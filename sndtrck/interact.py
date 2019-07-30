import sys
import operator as _op
from functools import lru_cache
import logging

import pyqtgraph as pg
from pyqtgraph import GraphicsObject, Point
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.functions as fn

from emlib.pitchtools import db2amp, f2n


from . import plotpyqtgraph as ppg
from .plotpyqtgraph import qtapp
from .synthesis import SpectrumPlayer
from .sinesynth import SineSynth, MultiSineSynth
from .spectralsurface import SpectralSurface
from .config import getconfig
from . import spectrum as _sp


logger = logging.getLogger("sndtrck")


def interact(sp, updaterate=0, block=False, ownapp=False):
    # type: (_sp.Spectrum, int, bool) -> SpectrumEditor
    """
    updaterate: the updaterate in Hz. Use 0 for configurable default (key='spectrumeditor.updaterate')
    block: if True, execution is blocked until the window is closed, after
                    which the spectrum-editor is itself closed
           if False, the interactive window is opened async and returns as 
                     soon as possible. You need to call "close" when finished
                     with the SpectrumEditor. Otherwise cleanup is done 
                     when exiting python (atexit)
    """
    if updaterate <= 0:
        updaterate = getconfig()['spectrumeditor.updaterate']
    with qtapp(ownapp):
        se = SpectrumEditor(sp, updaterate=updaterate)
        if block:
            se.wait_until_closed()
    return se


class Cursor(GraphicsObject):
    sigPlotChanged = QtCore.Signal(object)
    sigPositionChanged = QtCore.Signal(object)

    def __init__(self, pos):
        super().__init__()
        self.pos = pos
        self.pen = fn.mkPen((255, 255, 100), width=1)
        self.y1 = 5000
        self.picture = None
        self._boundingRect = None
        self._line = None
        
    def dataBounds(self, ax, *args, **kws):
        if ax == 0:
            return self.pos, self.pos
        return 0, self.y1

    def boundingRect(self):
        if self._boundingRect is not None:
            return self._boundingRect
        br = self.viewRect()
        if br is None:
            return QtCore.QRectF()
        
        px = self.pixelLength(direction=Point(1,0), ortho=True) or 0   # get pixel length orthog. to line
        w = 1 * px
        br.setLeft(-w)
        #br.setLeft(0)
        br.setRight(w)
        # br.setRight(0)
        
        br = br.normalized()
        self._boundingRect = br
        return br
    
    def paint(self, p, opt, widget, **args):
        p.setPen(self.pen)
        line = self._line
        if line is None:
            br = self.boundingRect()
            self._line = line = QtCore.QLineF(0.0, br.bottom(), 0.0, br.top())
        p.drawLine(line)
        
    def setPos(self, pos):
        if pos != self.pos:
            self.pos = pos
            # self.informViewBoundsChanged()
            # self.sigPlotChanged.emit(self)
            # self._boundingRect = None
            GraphicsObject.setPos(self, Point([self.pos, 0]))
            # self.sigPositionChanged.emit(self)


class _SpectrumWidget:

    def __init__(self, full=True):
        self.plotview = None
        self.partialsplot = None
        self.plotparams = {}
        self._scenepos = None
        self._refs = []
        self._vb = None
        
        if not full:
            print("******************* not full ******************* ")
            raise RuntimeError("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            self.plotview = pg.plot()
            self._fullwidget = False
            self._root = self.plotview
            self._win = None
            return

        labelwidth, valuewidth = 80, 70
        spinfontsize, labelfontsize = 10, 10
        self._fullwidget = True

        def change_fontsize(widget, size):
            font = widget.font()
            font.setPointSize(size)
            widget.setFont(font)

        def set_width(obj, width):
            obj.setMaximumWidth(width)
            obj.setMinimumWidth(width)

        def set_font_and_width(w, size, width):
            change_fontsize(w, size)
            set_width(w, width)

        def makelabel(txt):
            label = QtGui.QLabel(text=txt)
            set_font_and_width(label, labelfontsize, labelwidth)
            return label

        self._root = w = QtGui.QWidget()
        self._layout = layout = QtGui.QGridLayout()
        w.setLayout(layout)
        row = 0
        
        def addcheckbox(text:str) -> QtGui.QCheckBox:
            nonlocal row
            chk = QtGui.QCheckBox()
            chk.setTristate(False)
            layout.addWidget(makelabel(text), row, 0)
            layout.addWidget(chk, row, 1)
            row += 1
            return chk

        def addcombo(label:str, options:'List[str]') -> QtGui.QComboBox:
            nonlocal row
            combo = QtGui.QComboBox()
            for opt in options:
                combo.addItem(opt)
            layout.addWidget(makelabel(label), row, 0)
            layout.addWidget(combo, row, 1)
            row += 1
            return combo

        def addspin(label:str, value:float, step=0.1, decimals=1, maximum=999999, minimum=0):
            nonlocal row
            spinbox = QtGui.QDoubleSpinBox(value=value, singleStep=step, decimals=decimals)
            spinbox.setMaximum(maximum)
            spinbox.setMinimum(minimum)
            layout.addWidget(makelabel(label), row, 0)
            layout.addWidget(spinbox, row, 1)
            spinbox.setKeyboardTracking(False)
            set_font_and_width(spinbox, spinfontsize, valuewidth)
            row += 1
            return spinbox

        def addspinrange(label, value0, value1, step=0.1, decimals=3):
            nonlocal row
            spin0 = QtGui.QDoubleSpinBox(value=value0, singleStep=step, decimals=decimals)
            spin0.setKeyboardTracking(False)
            spin1 = QtGui.QDoubleSpinBox(value=value1, singleStep=step, decimals=decimals)
            spin1.setKeyboardTracking(False)
            layout.addWidget(makelabel(label), row, 0)
            layout.addWidget(spin0, row, 1)
            layout.addWidget(spin1, row+1, 1)
            set_font_and_width(spin0, spinfontsize, valuewidth)
            set_font_and_width(spin1, spinfontsize, valuewidth)
            row += 2
            return spin0, spin1

        cfg = getconfig()
        self.speed_spin = addspin("Speed", 1, step=0.01, decimals=2, maximum=50, minimum=-50)
        self.play             = addcheckbox("Play (Spc)")
        self.scrub            = addcheckbox("Scrub (0)")
        self.loop             = addcheckbox("Loop (l)")  
        self.filterchk        = addcheckbox("Filter (f)")
        self.nearest_partial  = addcheckbox("Nearest (p)")
        self.loudest_partials = addcheckbox("Loudest (k)")
        self.numloudest = addspin("Max. Partials", value=4, step=1, decimals=0)
        self.kind = addcombo("Kind", ["Amplitude", "Bandwidth"])
        self.exponent = addspin("Exponent", value=cfg["spectrumeditor.exp"], 
                                step=0.1, decimals=1)
        self.downsample = addspin("Downsample", cfg['spectrumeditor.downsample'], 
                                  step=1, decimals=0, minimum=1)
        self.bwfilter0, self.bwfilter1 = addspinrange("Min/Max Bw", 0, 1, step=0.001)
        self.gain = addspin("Gain", value=cfg['spectrumeditor.gain'])
        self.noisegain = addspin("Noise Gain", value=cfg['spectrumeditor.noisegain'],
                                 maximum=999)

        self.notesbtn = QtGui.QPushButton("Notes")
        layout.addWidget(self.notesbtn, row, 0)
        row += 1
        
        self.quitbtn = QtGui.QPushButton("Close")
        layout.addWidget(self.quitbtn, row, 0)
        row += 1
        
    def show(self) -> None:
        self._win = self._root.show()

    def _mouseMoved(self, ev) -> None:
        # snipped taken from https://stackoverflow.com/questions/35528198/returning-mouse-cursor-coordinates-in-pyqtgraph
        self._scenepos = ev[0]

    def mousepos(self):
        # return self.plotview.plotItem.vb.mapSceneToView(self._scenepos)
        return self._vb.mapSceneToView(self._scenepos)
        
    def plotspectrum(self,
                     s: _sp.Spectrum,
                     *,
                     downsample:int=-1,
                     alpha=-1,
                     linewidth:int=-1,
                     exp:float=-1,
                     numcolors:int=-1,
                     antialias=True,
                     background=None,
                     kind='amp') -> None:
        """ 
        downsample: if 2, 
        """
        config = getconfig()
        if downsample < 0: 
            downsample = config['spectrumeditor.downsample']
        if alpha < 0:      
            alpha = config['spectrumeditor.alpha']
        if linewidth < 0:  
            linewidth = config['spectrumeditor.linewidth']
        if exp < 0:        
            exp = config['spectrumeditor.exp']
        if numcolors < 0:
            numcolors = config['spectrumeditor.numcolors']
        oldopts = ppg.set_temp_options(background=background, useOpenGL=True)
        if self.plotview is None: 
            # self.plotview = ppg.SndtrckPlotWidget()
            self.plotview = pg.PlotWidget()
            self._layout.addWidget(self.plotview, 0, 2, 20, 1)
            self._refs.append(pg.SignalProxy(self.plotview.scene().sigMouseMoved, rateLimit=20, slot=self._mouseMoved))
            self._vb = self.plotview.plotItem.vb
            
        allpens = ppg.makepens(numcolors, alpha, linewidth)
        if self.partialsplot is None:
            self.plotview.setRange( QtCore.QRect(s.t0, 100, s.t1, 5000) )
            self.plotview.centralWidget.vb.setLimits(xMin=0, xMax=s.t1+0.1, yMin=0, yMax=22000)
        self.partialsplot = ppg.plotpartials(self.plotview, s, allpens, exp=exp, downsample=downsample,
                                             antialias=antialias, kind=kind, widget=self.partialsplot)
        pg.setConfigOptions(**oldopts)
        self.plotparams.update(
            downsample=downsample, alpha=alpha, linewidth=linewidth, exp=exp,
            numcolors=numcolors, background=background, kind=kind
        )

    def close(self):
        logger.debug("********************* SpectrumWidget.close")


def plotspectrum(s: _sp.Spectrum, *,
                 downsample=1, alpha:int=255, linewidth:int=2,
                 exp=1.0, numcolors:int=500, antialias=True,
                 background=None, kind:str='amp') -> None:
    sw = _SpectrumWidget(full=False)
    sw.plotspectrum(s, downsample=downsample, alpha=alpha, linewidth=linewidth,
                    exp=exp, numcolors=numcolors, antialias=antialias, 
                    background=background, kind=kind)
    return sw.plotview 


class SpectrumEditor:
    def __init__(self, sp: _sp.Spectrum, updaterate=12, numloudest:int=None) -> None:
        cfg = getconfig()
        self.spectrum = sp
        self.playing = False
        self.playheadpos = sp.t0
        self.ui = w = _SpectrumWidget()

        updateperiod_ms = int(1000 / updaterate)
        w.plotspectrum(sp)
        w.show()
        view = w.plotview
        plot = w.partialsplot

        self.view = view
        self._plot = plot    
        self._speed = 1
        self._editpos = sp.t0
        self._endpos = sp.t1
        self._looping = False
        self._t0 = sp.t0
        self._t1 = sp.t1
        self._sinesynth = None
        self._chordsynth = None
        self._surface = None
        self._tasks = {}
        self._state = {}
        self._num_loudest_partials = numloudest or cfg['spectrumeditor.numloudest']

        # Playhead
        self._playhead_last = -1
        
        self._cursor = Cursor(self.playheadpos)
        self._cursor.setZValue(100)    
        self.view.addItem(self._cursor, ignoreBounds=True)

        self._cursor_update_visibility()
        
        def on_poschange(pos, self=self):
            self.playheadpos = pos
        
        self.player = SpectrumPlayer(
            sp, autoplay=False, 
            looping=self._looping,
            start=sp.t0, end=sp.t1, 
            speed=self._speed,
            on_play=lambda state:self.play(bool(state), _propagate=False),
            on_pos=on_poschange,
            oscfreq=updaterate
        )    

        # Edithead
        self._editcursor = pg.InfiniteLine(
            movable=True, 
            pos=self.playheadpos,
            bounds=(sp.t0, sp.t1), 
            pen=pg.mkPen("#56ff7688", width=2))
        self._editcursor.setZValue(20)
        self.view.addItem(self._editcursor)

        self._editcursor.sigPositionChanged.connect(self._editcursor_changed)

        # Endloop
        self._endposcursor = pg.InfiniteLine(
            movable=True, pos=self._endpos,
            bounds=(sp.t0, sp.t1), pen=pg.mkPen("#5676ff88", width=2))
        self.view.addItem(self._endposcursor)
        self._endposcursor.sigPositionChanged.connect(self._endposcursor_changed)
        self._endposcursor.setZValue(20)
        if not self._looping:
            self._endposcursor.hide()
        
        def newScatterCursor(anchor=(-0.3, 1.2)):
            cursor = pg.ScatterPlotItem()
            cursor.hide()
            cursortxt = pg.TextItem(text="--", anchor=anchor, fill="#00000066")
            cursortxt.setZValue(10)
            cursortxt.hide()
            self.view.addItem(cursor)
            self.view.addItem(cursortxt)
            return cursor, cursortxt

        self._synthcursor, self._synthcursortxt = newScatterCursor()
        self._chordcursor, self._chordcursortxt = newScatterCursor(anchor=(0.5, 1.5))
        
        # Timer
        self._timer = timer = QtCore.QTimer(self.view)
        timer.timeout.connect(self._update)
        timer.start(updateperiod_ms)

        # Filter
        self._filter_active = False
        self._filter_range = (300, 1000)
        self._filterwidget = pg.LinearRegionItem(
            values=self._filter_range, 
            orientation=pg.LinearRegionItem.Horizontal, 
            movable=True)
        self._filterwidget.setZValue(5)
        if not self._filter_active:
            self._filterwidget.hide()
        self._filterwidget.sigRegionChanged.connect(self._filter_changed)
        self.view.addItem(self._filterwidget)
        
        self._shortcuts = []
        self._setup_keyboard()

        self._set_callbacks()
        self._plot.setFocus()


        # self.ui.execloop()

    def _set_callbacks(self) -> None:
        self.ui.speed_spin.valueChanged.connect(self.set_speed)
        self.ui.nearest_partial.stateChanged.connect(lambda state: self._synth_nearest_partial(bool(state)))
        self.ui.quitbtn.clicked.connect(self.close)
        self.ui.loudest_partials.stateChanged.connect(lambda state: self._synth_loudest_partials(bool(state)))
        self.ui.numloudest.valueChanged.connect(self.set_numloudest_partials)
        self.ui.notesbtn.clicked.connect(lambda:self.show_chord(self.playheadpos))
        self.ui.filterchk.stateChanged.connect(self.set_filter)
        self.ui.play.stateChanged.connect(lambda state:self.play(state > 0))
        self.ui.loop.stateChanged.connect(lambda state:self.set_looping(state > 0))
        self.ui.scrub.stateChanged.connect(lambda state:self.set_scrub(state > 0))
        self.ui.kind.currentIndexChanged.connect(lambda idx:self.set_kind(("amp", "bw")[idx]))
        self.ui.exponent.valueChanged.connect(self.set_exp)
        self.ui.gain.valueChanged.connect(lambda gain:self.player.set_gain(gain=gain))  
        self.ui.noisegain.valueChanged.connect(lambda gain:self.player.set_gain(noisegain=gain))
        self.ui.bwfilter0.valueChanged.connect(lambda x:self.set_bwfilter(x, -1))
        self.ui.bwfilter1.valueChanged.connect(lambda x:self.set_bwfilter(-1, x))
        self.ui.downsample.valueChanged.connect(lambda x:self.set_downsample(int(x)))

    def _editcursor_changed(self) -> None:
        x, y = self._editcursor.getPos()
        self.set_editpos(x, 0.05)

    def _filter_changed(self) -> None:
        f0, f1 = self._filterwidget.getRegion()
        self._filter_range = (f0, f1)
        self.set_filter(self._filter_active, f0, f1)

    def _endposcursor_changed(self) -> None:
        x, y = self._endposcursor.getPos()
        self.set_endpos(x)

    def _setup_keyboard(self):

        def setkey(keyseq, func):
            k = pg.QtGui.QKeySequence(keyseq)
            shortcut = pg.QtGui.QShortcut(k, self.view, func)
            self._shortcuts.append(shortcut)

        setkey("Space", lambda self=self:self.ui.play.setChecked(not self.playing))                                
        setkey("l", lambda:self.ui.loop.setChecked(not self._looping))
        setkey("b", lambda:self.set_editpos(self.ui.mousepos().x()))
        setkey("e", lambda:self.set_endpos(self.ui.mousepos().x()) or self.ui.loop.setChecked(True))
        setkey("f", lambda:self.ui.filterchk.setCheckState(not self._filter_active))
        setkey("p", lambda:self.ui.nearest_partial.setCheckState(not self._state.get('synth_nearest', False)))
        setkey("0", lambda:self.ui.scrub.setChecked(not self._state.get('tempspeed', False)))
        setkey("k", lambda:self.ui.loudest_partials.setChecked(not self._state.get('synth_loudest', False)))
        setkey("n", self.show_chord)
        setkey("Alt+q", self.close)

    def set_scrub(self, state):
        self._set_speed0(state)

    def set_synth_loudest(self, state=None):
        """
        True/False to turn it on/off, None to toggle
        """
        if state is None:
            state = not self._state.get('synth_loudest', False)
        self._synth_loudest_partials(state)
        self.ui.loudest_partials.setChecked(state)
        
    def set_playhead(self, x:float, dur:float) -> None:
        self._cursor.show()
        self.player.set_playhead(x, dur)

    def set_editpos(self, x:float, dur=0.01) -> None:
        self._editpos = x
        self.player.set_position(x, dur)
        self._editcursor.setPos(x)
        if x > self._endpos:
            self.set_endpos(x + (self._endpos - self._editpos))
        
    def set_endpos(self, x:float) -> None:
        self._endpos = x = max(self._editpos, min(x, self._t1))
        self.player.set_selection_end(x)
        self._endposcursor.setPos(x)

    def set_kind(self, kind:str) -> None:
        self.ui.plotparams['kind'] = kind
        self.ui.plotspectrum(self.spectrum, **self.ui.plotparams)

    def set_exp(self, exp):
        self.ui.plotparams['exp'] = exp
        self.ui.plotspectrum(self.spectrum, **self.ui.plotparams)

    def set_bwfilter(self, bw0, bw1):
        self.player.set_bwfilter(bw0, bw1)

    def set_downsample(self, n:int) -> None:
        self.ui.plotparams['downsample'] = n
        self.ui.plotspectrum(self.spectrum, **self.ui.plotparams)

    def set_gain(self, gain:float) -> None:
        self.player.set_gain(gain)

    def set_noisegain(self, noisegain:float) -> None:
        self.player.set_noisegain(noisegain)

    def _set_speed0(self, state):
        if state:
            self._state['tempspeed'] = True
            self._state['tempspeed.lastspeed'] = self._speed
            self.set_speed(0)
            self.set_editpos(self.ui.mousepos().x())
            self.set_looping(False)
            self.play(True)
        else:
            lastspeed = self._state.get('tempspeed.lastspeed', 1)
            self.set_speed(lastspeed)
            self._state['tempspeed'] = False
            self.play(False)

    def _synth_mousepos(self, state):
        self._state['synth_mouse'] = state

        def updatesine():
            synth = self._sinesynth
            mousey = self.ui.mousepos().y()
            synth.setFreq(mousey)

        if state:
            self._sinesynth = SineSynth(freq=self.ui.mousepos().y(), amp=0.75)
            self._addtask("sinesynth", updatesine)
        else:
            self._removetask("sinesynth")
            self._sinesynth.stop()
            self._sinesynth = None

    def show_chord(self, t:float=None, method:str=None):
        """
        t: time at which to get the chord
           If None, use time at mouse position
        method: as passed to a music21 stream .show method
        """
        if method is None:
            config = getconfig()
            method = config['show.show_chord.method']

        if t is None:
            t = self.ui.mousepos().x()
        numnotes = self._num_loudest_partials
        chord = self.spectrum.chord_at(t, maxnotes=numnotes)
        chord.asmusic21().show(method)

    def _synth_loudest_partials(self, state:bool, minamp_db=-60):
        if not state:
            synthon = self._state['synth_loudest']
            if not synthon:
                logger.debug("Asked to stop 'synth_loudest_partials', but it was not on!")
                return 
            self._chordcursor.hide()
            self._chordcursortxt.hide()
            self._removetask("loudest")
            self._chordsynth.fadeout()

            def stop(synth=self._chordsynth, state=self._state):
                synth.stop()
                state['synth_loudest'] = False
            
            QtCore.QTimer.singleShot(self._chordsynth.fadetime * 1000 + 50, stop)
            self._state['synth_loudest.lasttime'] = -1
            self._state['synth_loudest.lastdraw'] = -1
            return

        if self._state.get('synth_loudest', False):
            logger.debug("Asked to start 'synth_loudest_partials', but it was already on!")
            return 

        self._state['synth_loudest'] = True
        maxpartials = self._num_loudest_partials
        from heapq import nlargest

        @lru_cache(maxsize=1000)
        def partials_at(t):
            return self.spectrum.partials_at(t)

        @lru_cache(maxsize=1000)
        def loudest(t, maxpartials, minamp, filteron, freq0, freq1):
            trnd = round(t*200)/200.  # round to 0.005 for cache
            partials = partials_at(trnd)
            if not partials:
                return
            data = []
            for p in partials:
                amp = p.amp(t)
                if amp > minamp:
                    freq = p.freq(t)
                    bw = p.bw(t)
                    if filteron:
                        if freq0 <= freq < freq1:
                            data.append((amp, freq, bw))
                    else:
                        data.append((amp, freq, bw))
            if not data:
                return
            best = nlargest(maxpartials, data)
            best.sort(key=_op.itemgetter(1))
            return best

        def loudest_task(self=self, minamp=db2amp(minamp_db)):
            pl = self._plot
            mousepos = self.ui.mousepos()
            time = mousepos.x()
            lasttime = self._state.get('synth_loudest.lasttime', -1)
            if abs(time - lasttime) < 0.001 or time <= 0:
                return
            best = loudest(time, maxpartials, minamp, self._filter_active, *self._filter_range)
            if not best:
                return 
            
            freqs = [row[1] for row in best]
            synth = self._chordsynth
            lastdraw = self._state.get('synth_loudest.lastdraw', 0)
            if abs(time - lastdraw) > 0.005:
                self._chordcursor.setData([time]*len(freqs), freqs)
                self._state['synth_loudest.lastdraw'] = time
            for i, row in enumerate(best):
                synth.setOsc(i, row[1], row[0], row[2])
            self._state['synth_loudest.lasttime'] = time

            self._chordcursortxt.setPos(time, freqs[-1])
            txt = "%.3f" % time
            self._chordcursortxt.setText(txt)
            
        self._chordcursor.show()
        self._chordcursortxt.show()
        porttime = getconfig()['sinesynth.porttime']
        self._chordsynth = MultiSineSynth(freqs=[1000]*maxpartials, 
                                          amps=[0]*maxpartials, 
                                          bws=[0]*maxpartials, 
                                          porttime=porttime)
        self._addtask("loudest", loudest_task)
    
    def _synth_nearest_partial(self, state:bool, margin_hz=50, minamp_db=-60, gain=2.0):
        currstate = self._state.get('synth_nearest', False)
        if state == currstate:
            return
        
        if not state:
            self._synthcursor.hide()
            self._synthcursortxt.hide()
            self._removetask("nearest")
            fadetime = 0.2
            self._sinesynth.fadeout(fadetime)

            def stop(synth=self._sinesynth, state=self._state):
                synth.stop()
                state['synth_nearest'] = False
            
            QtCore.QTimer.singleShot(fadetime * 1000 + 50, stop)
            return 

        minamp = db2amp(minamp_db)

        def update(self=self):
            synth = self._sinesynth
            pl = self._plot
            mousepos = self.ui.mousepos()
            time = mousepos.x()
            mousefreq = mousepos.y()
            freq, amp = self._surface.nearest(time, mousefreq)
            if amp < minamp:
                return
            if abs(freq - mousefreq) > margin_hz:
                amp = 0
            if freq <= 0:
                return
            synth.setOsc(0, freq, amp)
            self._synthcursor.setData([time], [freq])
            self._synthcursortxt.setPos(time, freq)
            txt = "%d Hz  %s  %.3f" % (int(freq), f2n(freq), time)
            self._synthcursortxt.setText(txt)
            
        self._synthcursor.show()
        self._synthcursortxt.show()
        self._surface = SpectralSurface(self.spectrum, decay=0.01)
        self._sinesynth = SineSynth(freq=self.ui.mousepos().y(), amp=0, freqport=0.2, 
                                    gain=gain)
        self._addtask("nearest", update)   
        self._state['synth_nearest'] = True
     
        
    def _addtask(self, name, func):
        if name in self._tasks:
            raise KeyError("A task with that name already exists!")
        self._tasks[name] = func

    def _removetask(self, name):
        self._tasks.pop(name)

    def _update(self):
        pos = self.playheadpos
        if pos != self._playhead_last:
            self._cursor.setPos(pos)
            self._playhead_last = pos
        if self._tasks:
            for task in self._tasks.values():
                task()

    def set_speed(self, speed:float) -> None:
        self._speed = speed
        self.player.set_speed(speed)
        self.ui.speed_spin.setValue(speed)
        self._cursor_update_visibility()

    def set_numloudest_partials(self, num:int) -> None:
        self._num_loudest_partials = int(num)
        #if self._state.get('synth_loudest', False):
        #    self._synth_loudest_partials(False)
        #    self._synth_loudest_partials(True)
        
    def _cursor_update_visibility(self):

        if self.playing and self._speed != 0:
            self._cursor.show()
        else:
            self._cursor.hide()

    def set_looping(self, state:bool) -> None:
        self._looping = state
        self.player.set_looping(state)
        if self._looping:
            self._endposcursor.show()
            self.player.set_selection_end(self._endpos)
        else:
            self._endposcursor.hide()
            self.player.set_selection_end(self._t1)

    def set_filter(self, active:bool, freq0=-1, freq1=-1):
        oldfreq0, oldfreq1 = self._filter_range
        freq0 = freq0 if freq0 > 0 else oldfreq0
        freq1 = freq1 if freq1 > 0 else oldfreq1
        self._filter_range = (freq0, freq1)
        self._filter_active = active
        self.player.set_filter(active, freq0, freq1)
        if active:
            self._filterwidget.show()
        else:
            self._filterwidget.hide()

    def close(self):
        self.spectrum = None
        self._surface = None
        self._plot = None
        if self._sinesynth is not None:
            self._sinesynth.stop()
            self._sinesynth = None
        if self._chordsynth is not None:
            self._chordsynth.stop()
            self._chordsynth = None
        if self._tasks:
            for task in list(self._tasks.keys()):
                self._removetask(task)
        self._timer.stop()
        self.player.exit()
        self.view.window().close()
        self.view.setParent(None)
        self.ui.close()
        
    def wait_until_closed(self):
        """
        Blocks until the window is closed, then closes itself
        """
        _startloop()
        self.close()

    def play(self, state=None, loop=None, _propagate=True):
        """
        state: True  -> begin playing, or do nothing if already playing
               False -> stop playing or do nothing if already stopped
               None  -> toggle playing
        """
        if state is None:
            state = not self.playing
        assert isinstance(state, bool), f"state should be bool, but got {state}"
        if state != self.playing:
            self.playing = state
            self._cursor_update_visibility()
            if _propagate:
                if loop is not None:
                    self.ui.loop.setChecked(loop)
                self.player.play(state)
        self.ui.play.setChecked(state)


def _startloop():
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        logger.debug("startloop: starting loop")
        QtGui.QApplication.instance().exec_()
