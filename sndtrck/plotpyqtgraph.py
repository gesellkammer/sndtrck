import sys
from math import inf
from contextlib import contextmanager

from pyqtgraph import GraphicsObject, getConfigOption, Point
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.functions as fn
import pyqtgraph as pg

from bpf4 import bpf
from emlib.pitchtoolsnp import amp2db_np, f2m_np
from emlib import lib
from numpyx import minmax1d
import logging
from functools import lru_cache


logger = logging.getLogger("sndtrck")

# try:
#     from pyqtgraph.Qt import QtOpenGL
#     import OpenGL.GL as gl
#     HAVE_OPENGL = True
# except ImportError:
#     HAVE_OPENGL = False

HAVE_OPENGL = False

pg.setConfigOptions(
    background=(0, 0, 0),
    useOpenGL=True,
    enableExperimental=True,
    antialias=True
)


class MultiColouredLine(GraphicsObject):

    def __init__(self, antialias=True):
        GraphicsObject.__init__(self)
        self.data = []
        self.x0, self.x1, self.y0, self.y1 = inf, -inf, inf, -inf
        self.picture = None
        self._boundingRect = None
        self._numcalls = 0
        self._t1 = 0
        self._t0 = 0
        self._mousex = 0
        self._mousey = 0
        self._antialias = antialias

    def reset(self):
        self.data = []
        self.x0, self.x1, self.y0, self.y1 = inf, -inf, inf, -inf
        self._boundingRect = None
        self.picture = None

    def add(self, x, y, pens):
        self.data.append((x, y, pens))
        x0 = min(x[0], self.x0)
        x1 = max(x[-1], self.x1)
        miny, maxy = minmax1d(y)
        y0 = min(miny, self.y0)
        y1 = max(maxy, self.y1)
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def updateBounds(self):
        y0 = float("inf")
        y1 = -y0
        x0 = y0
        x1 = y1
        for X, Y, pens in self.data:
            miny, maxy = minmax1d(Y)
            if miny < y0:
                y0 = miny
            if maxy > y1:
                y1 = maxy 
            minx, maxx = minmax1d(X)
            if minx < x0:
                x0 = minx
            if maxx > x1:
                x1 = maxx
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def dataBounds(self, ax, *args, **kws):
        if ax == 0:
            return (self.x0, self.x1)
        return (self.y0, self.y1)

    def generatePicture(self):
        if not self.data:
            return
        picture = QtGui.QPicture()
        p = QtGui.QPainter(picture)
        setPen = p.setPen
        drawLine = p.drawLine
        lastpen = None
        try:
            P = QtCore.QPointF
            for X, Y, pens in self.data:
                numpoints = len(X)
                p0 = P(X[0], Y[0])
                for i in range(numpoints - 1):
                    i1 = i + 1
                    p1 = P(X[i1], Y[i1])
                    pen = pens[i]
                    if pen is not lastpen:
                        setPen(pen)
                        lastpen = pen
                    drawLine(p0, p1)
                    p0 = p1
        finally:
            p.end()
        return picture

    def paint(self, p, opt, widget, **args):
        # if HAVE_OPENGL and getConfigOption('enableExperimental'):
        #     if isinstance(widget, QtOpenGL.QGLWidget):
        #         self.paintGL(p, opt, widget)
        #         return
        if self.picture is None:
            self.picture = self.generatePicture()
        if self._antialias:
            p.setRenderHint(p.Antialiasing)
        self.picture.play(p)

    def boundingRect(self):
        if self._boundingRect is not None:
            return self._boundingRect
        (xmn, xmx) = self.dataBounds(ax=0)
        (ymn, ymx) = self.dataBounds(ax=1)
        if xmn is None or xmx is None:
            xmn, xmx = 0, 0
        if ymn is None or ymx is None:
            ymn, ymx = 0, 0
        px = py = 0.0
        # pxPad = self.pixelPadding()
        pxPad = 1
        if pxPad > 0:
            # determine length of pixel in local x, y directions
            px, py = self.pixelVectors()
            try:
                px = 0 if px is None else px.length()
            except OverflowError:
                px = 0
            try:
                py = 0 if py is None else py.length()
            except OverflowError:
                py = 0
            # return bounds expanded by pixel size
            px *= pxPad
            py *= pxPad
        self._boundingRect = QtCore.QRectF(xmn-px, ymn-py, (2*px)+xmx-xmn, (2*py)+ymx-ymn)
        return self._boundingRect


def _colormapGray(x, alpha):
    x2 = int(x * 255)
    return (x2, x2, x2, alpha)


def _colormapHeat(x, alpha):
    # x should be 0-1
    j = int(x*4)
    if j == 0:
        x *= 4
        r = x*69
        g = x*25
        b = x*94
    elif j == 1:
        x = (x-0.25) * 4
        r = 60+x*(216-60)
        g = 25+x*(82-25)
        b = 94+x*(50-94)
    elif j == 2:
        x = (x-0.5)*4
        r = 216+x*(253-216)
        g = 82+x*(162-82)
        b = 50+x*(28-50)
    elif j == 3:
        x = (x-0.75)*4
        r = 253+x*(255-253)
        g = 162+x*(255-162)
        b = 28+x*(255-28)
    elif j >= 4:
        r, g, b = (255, 255, 255)
    return int(r), int(g), int(b), alpha
  

_db2lin = bpf.linear(-120, 0.001, -60, 0.1, -12, 0.75, -3, 0.85, 0, 1)
_bw2lin = bpf.linear(0, 0.1, 
                     0.0001, 0.2, 
                     0.9, 0.6, 
                     0.95, 0.8, 
                     1, 0.99)
# _bw2lin = bpf.linear(0, 0.99, 0.5, 0.5, 1, 0.1)

_ownapp = None

@contextmanager
def qtapp(ownapp=False):
    global _ownapp
    if ownapp:
        app = QtGui.QApplication([])
        # Keep a reference here
        _ownapp = app
    else:
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtGui.QApplication([])
    yield app
    if ownapp or not lib.ipython_qt_eventloop_started():
        print("*************************************** starting eventloop")
        app.exec_()

def plotpartials_(v, partials, allpens, *, widget=None, exp=1, downsample=1, antialias=True, kind='amp', pitchmode='freq'):
    numpens = len(allpens)
    getZ = lambda partial:partial.amps
    curve = _db2lin

    def transform(arr):
        arr = amp2db_np(arr)
        return curve.map(arr, out=arr)

    for partial in partials:
        X = partial.times
        Y = partial.freqs
        Z = getZ(partial)
        if downsample > 1:
            X = X[::downsample]
            Y = Y[::downsample]
            Z = Z[::downsample]
        Z = transform(Z)
        if exp != 1:
            Z **= exp
        colors = (Z[1:] + Z[:-1]) * (0.5 * numpens)
        # colors = np.floor(colors, out=colors)
        colors = colors.astype(int)
        colors.clip(0, numpens-1, out=colors)
        pens = [allpens[col] for col in colors]
        if pitchmode == 'note':
            Y = f2m_np(Y, out=Y)
        line = MultiColouredLine()
        line.add(X, Y, pens)
        v.addItem(line, ignoreBounds=True)
    return line


def plotpartials(v, partials, allpens, *, widget=None,
                 exp=1, downsample=1, 
                 antialias=True, kind='amp', pitchmode='freq'):
    """
    v: a View as returned by, for example, pg.plot()
    partials: a seq of Partials
    allpens: a list of pens used to paint each color
    """
    logger.debug(f"~~~~~~~~~~~~~~~~~~~~~~~~~~ plotpartials: downsample={downsample}, kind={kind}")
    if widget is None:
        m = MultiColouredLine(antialias=antialias)
    else:
        m = widget
        m.reset()
    numpens = len(allpens)
    if kind == 'amp':
        curve = _db2lin
        getZ = lambda partial:partial.amps
        
        def transform(arr):
            arr = amp2db_np(arr)
            return curve.map(arr, out=arr)
    elif kind == 'bw':
        curve = _bw2lin
        getZ = lambda partial:partial.bws
        transform = lambda arr:curve.map(arr)
    else:
        raise ValueError(f"kind should be 'amp' or 'bw', got {kind}")

    for partial in partials:
        X = partial.times
        Y = partial.freqs
        Z = getZ(partial)
        if downsample > 1:
            X = X[::downsample]
            Y = Y[::downsample]
            Z = Z[::downsample]
        Z = transform(Z)
        if exp != 1:
            Z **= exp
        colors = (Z[1:] + Z[:-1]) * (0.5 * numpens)
        # colors = np.floor(colors, out=colors)
        colors = colors.astype(int)
        colors.clip(0, numpens-1, out=colors)
        pens = [allpens[col] for col in colors]
        if pitchmode == 'note':
            Y = f2m_np(Y, out=Y)
        m.add(X, Y, pens)
    v.addItem(m, ignoreBounds=True)
    return m


@lru_cache(maxsize=50)
def makepens(numcolors, alpha, linewidth):
    colormap = _colormapHeat
    return [fn.mkPen(color=colormap(i/numcolors, alpha), width=linewidth) 
            for i in range(numcolors)]


def set_temp_options(**options) -> dict:
    oldopts = {key:pg.getConfigOption(key) for key in options}
    validoptions = {k:v for k, v in options.items() if v is not None}
    pg.setConfigOptions(**validoptions)
    return oldopts


def plotspectrum(s, *, downsample=1, alpha:int=255, linewidth:int=2, 
                 exp=1, numcolors=500, antialias=True,
                 background=None, kind='amp', pitchmode='freq'):
    """
    kind: amp or bw
    pitchmode: freq or note

    s: the spectrum to plot
    """
    from . import spectrum
    assert isinstance(s, spectrum.Spectrum)
    assert pitchmode in ("freq", "note")
    oldopts = set_temp_options(background=background)
    with qtapp():
        v = pg.plot(title="plot")
        allpens = makepens(numcolors, alpha, linewidth)
        plot = plotpartials(v, s, allpens, exp=exp, downsample=downsample, 
                            antialias=antialias, kind=kind, pitchmode=pitchmode)
        rect = QtCore.QRect(s.t0, 0, s.t1, 6000)
        v.setRange(rect)
        v.centralWidget.vb.setLimits(xMin=0, xMax=s.t1+0.1, yMin=0, yMax=24000)
        pg.setConfigOptions(**oldopts)
        # v.scene().sigMouseMoved.connect(lambda *args:print("***********", args))
        # proxy = pg.SignalProxy(v.scene().sigMouseMoved, rateLimit=40, slot=lambda *args: print(">>>>", args))
    return v, plot


def startloop():
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        logger.info("startloop: starting Qt loop")
        QtGui.QApplication.instance().exec_()
