from __future__ import absolute_import
from matplotlib import pyplot as plt
import bpf4 as bpf
from emlib.pitch import *
from . import partial as _partial
from . import log
from typing import Any

logger = log.get_logger()

config = {
    'plotbackends': ['pyqtgraph', 'pyplot'],
}


def _get_plot_func(backendname):
    backends = {
        'pyqtgraph': plot_pyqtgraph,
        'pyplot'   : plot_pyplot
    }
    func = backends.get(backendname)
    if func is None:
        raise KeyError(f"Backend {backendname} not found!")
    return func


def plot(spectrum, linewidth=2, downsample=1, antialias=True, exp=1, 
         showpoints=False, **kws):
    """
    downsample: 1 plots all breakpoints, 2 plots 1/2, 3 plots 1/3 
                of all breakpoints, etc.
    antialias: depending on the backend, can be turned off to
               have better performance for very large plots
    exp: apply an exponential factor (amps ** exp) over the amplitude 
         to control contrast. 
         * If exp == 0.5, week breakpoints will be more visible
         * If exp == 2, only very strong breakpoints will be visible
    """
    backendnames = config['plotbackends']
    for backendname in backendnames:
        func = _get_plot_func(backendname)
        logger.debug(f"plot: using backend {backendname}")
        success = func(spectrum, downsample=downsample, antialias=antialias, exp=exp, 
                       linewidth=linewidth, showpoints=showpoints, **kws)
        if success:
            break
    else:
        raise RuntimeError(f"No backend available. Backends tried: {backendnames}")


def plotdraft(spectrum, linewidth=2, exp=1):
    return plot_pyqtgraph(spectrum, linewidth=linewidth, exp=exp, downsample=2, 
                          antialias=False, showpoints=False, numcolors=32)


def plot_pyplot(partials, size=20, alpha=0.5, downsample=1, **kws):
    fig, ax = plt.subplots(1, 1)
    for p in partials:
        _pyplot_plot_partial(p, ax, size, alpha, downsample=downsample)
    plt.show()
    return True


def _pyplot_plot_partial(partial, ax=None, size=20, alpha=0.5, downsample=1):
    # type: (_partial.Partial, Any, float, float) -> Any
    if ax is None:
        fig, ax = plt.subplots(1, 1)    
    X = partial.times     # type: np.ndarray
    Y = partial.freqs     # type: np.ndarray
    amps = partial.amps   # type: np.ndarray
    if downsample > 1:
        X = X[::downsample]
        Y = Y[::downsample]
        amps = amps[::downsample]
    dbs = amp2db_np(amps)
    colorcurve = bpf.linear(-90, 0.8, -30, 0.3, -6, 0.1, 0, 0)
    C = colorcurve.map(dbs)
    plt.gray()
    linecol = colorcurve(-70)
    ax.plot(X, Y, color=(linecol, linecol, linecol, 0.5))
    ax.scatter(X, Y, size, c=C, linewidth=0, alpha=alpha)
    return ax


def plot_pyqtgraph(spectrum, *, alpha=1, linewidth=2, 
                   exp=1, numcolors=500, downsample=1,
                   antialias=False, showpoints=False) -> bool:
    try:
        from . import plotpyqtgraph
    except ImportError:
        return False

    if showpoints:
        alpha = 0.5
    plotpyqtgraph.plotspectrum(spectrum, alpha=int(alpha*255), linewidth=linewidth, 
                               exp=exp, numcolors=numcolors, downsample=downsample,
                               antialias=antialias)
    plotpyqtgraph.startloop()
    return True