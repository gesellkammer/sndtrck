from pyqtgraph import GraphicsObject, getConfigOption
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.functions as fn
import numpy as np
import pyqtgraph as pg
from bpf4 import bpf
from emlib.pitch import amp2db_np
from math import inf
import sys
from .accel import minmax1d

try:
    from pyqtgraph.Qt import QtOpenGL
    import OpenGL.GL as gl
    HAVE_OPENGL = True
except:
    HAVE_OPENGL = False

pg.setConfigOptions(
    background=(0, 0, 0),
    useOpenGL=False,
    enableExperimental=False,
    antialias=True
)


class MultiColouredLine(GraphicsObject):
    def __init__(self):
        GraphicsObject.__init__(self)
        self.data = []
        self.x0, self.x1, self.y0, self.y1 = inf, -inf, inf, -inf
        self.picture = None
        self._boundingRect = None

    def add(self, x, y, pens):
        self.data.append((x, y, pens))
        x0 = min(x[0], self.x0)
        x1 = max(x[-1], self.x1)
        miny, maxy = minmax1d(y)
        y0 = min(miny, self.y0)
        y1 = max(maxy, self.y1)
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def dataBounds(self, ax, *args, **kws):
        if ax == 0:
            return (self.x0, self.x1)
        return (self.y0, self.y1)

    def generatePicture(self):
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
        #if HAVE_OPENGL and getConfigOption('enableExperimental'):
        #    if isinstance(widget, QtOpenGL.QGLWidget):
        #        self.paintGL(p, opt, widget)
        #        return

        if self.picture is None:
            self.picture = self.generatePicture()
        if getConfigOption('antialias') is True:
            p.setRenderHint(p.Antialiasing)
        self.picture.play(p)

    def paintGL(self, p, opt, widget):
        view = self.getViewBox()
        if view is None:
            return 
        p.beginNativePainting()
        
        ## set clipping viewport
        rect = view.mapRectToItem(self, view.boundingRect())
        #gl.glViewport(int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))
        
        #gl.glTranslate(-rect.x(), -rect.y(), 0)
        if True:
            gl.glEnable(gl.GL_STENCIL_TEST)
            gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE) # disable drawing to frame buffer
            gl.glDepthMask(gl.GL_FALSE)  # disable drawing to depth buffer
            gl.glStencilFunc(gl.GL_NEVER, 1, 0xFF)  
            gl.glStencilOp(gl.GL_REPLACE, gl.GL_KEEP, gl.GL_KEEP)  
            
            ## draw stencil pattern
            gl.glStencilMask(0xFF)
            gl.glClear(gl.GL_STENCIL_BUFFER_BIT)
            gl.glBegin(gl.GL_TRIANGLES)
            gl.glVertex2f(rect.x(), rect.y())
            gl.glVertex2f(rect.x()+rect.width(), rect.y())
            gl.glVertex2f(rect.x(), rect.y()+rect.height())
            gl.glVertex2f(rect.x()+rect.width(), rect.y()+rect.height())
            gl.glVertex2f(rect.x()+rect.width(), rect.y())
            gl.glVertex2f(rect.x(), rect.y()+rect.height())
            gl.glEnd()
                       
            gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glStencilMask(0x00)
            gl.glStencilFunc(gl.GL_EQUAL, 1, 0xFF)
            
        try:
            # x, y = self.getData()
            try:
                poss = self._poss
            except:
                self._poss = poss = [(np.vstack([x, y]).T.copy(), pens) for x, y, pens in self.data]
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            try:
                for pos, pens in poss: 
                    gl.glVertexPointerf(pos)
                    # pen = fn.mkPen(self.opts['pen'])
                    pen = fn.mkPen((255, 255, 255), width=4)
                    color = pen.color()
                    gl.glColor4f(color.red()/255., color.green()/255., color.blue()/255., color.alpha()/255.)
                    width = pen.width()
                    if pen.isCosmetic() and width < 1:
                        width = 1
                    gl.glPointSize(width)
                    gl.glEnable(gl.GL_LINE_SMOOTH)
                    gl.glEnable(gl.GL_BLEND)
                    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                    gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
                    gl.glDrawArrays(gl.GL_LINE_STRIP, 0, int(pos.size / pos.shape[-1]))
            finally:
                gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        finally:
            p.endNativePainting()
        return

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


def _getview(plotview=True):
    if not plotview:
        w = pg.GraphicsWindow()
        w.useOpenGL(True)    
        w.setWindowTitle('pyqtgraph example: GraphItem')
        v = w.addViewBox()
        v.setAspectLocked()
    else:
        v = pg.plot()
    return v


def _plotpartials(v, partials, allpens, exp=1, downsample=1):
    m = MultiColouredLine()
    curve = bpf.linear(-120, 0.001, -60, 0.1, -12, 0.75, -3, 0.85, 0, 1)
    numpens = len(allpens)
    for partial in partials:
        X = partial.times
        Y = partial.freqs
        amps = partial.amps
        if downsample > 1:
            X = X[::downsample]
            Y = Y[::downsample]
            amps = amps[::downsample]
        # amp0 = 0.005
        # amps = (amps * (1 - amp0) + amp0)
        dbs = amp2db_np(amps)
        amps = curve.map(dbs) ** exp 
        colors = amps * numpens
        pens = [allpens[int(col)] for col in colors]
        m.add(X, Y, pens)
    v.addItem(m, ignoreBounds=True)


def plotspectrum(s, *, downsample=1, alpha:int=255, linewidth:int=2, 
                 exp=1, numcolors=500, antialias=True) -> None:
    pg.setConfigOptions(antialias=antialias)
    v = _getview()
    colormap = _colormapHeat
    allpens = [fn.mkPen(color=colormap(i/numcolors, alpha), width=linewidth) 
               for i in range(numcolors)]
    _plotpartials(v, s, allpens, exp, downsample=downsample)
    rect = QtCore.QRect(s.t0, 20, s.t1, 6000)
    v.setRange(rect)
    

def startloop():
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
