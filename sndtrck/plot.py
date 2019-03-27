import bpf4 as bpf
import logging
from emlib.pitchtoolsnp import amp2db_np
from . import partial as _partial
from .config import getconfig
from . import typehints as t


logger = logging.getLogger("sndtrck")


def plot(spectrum, linewidth=2, downsample=1, antialias=True, exp=1, 
         showpoints=False, kind='amp', pitchmode='freq', 
         backend=None, **kws):
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
    config = getconfig()
    backendname = backend or config['plot.backend']
    if backendname == 'pyplot':
        func = plot_pyplot
    elif backendname == 'pyqtgraph':
        func = plot_pyqtgraph
    else:
        raise KeyError(f"Backend {backendname} not found!")
    logger.debug(f"plot: using backend {backendname}")
    if showpoints and downsample > 1:
        if getconfig()['plot.showpoints_disables_downsample']:
            logger.debug("plot: downsampling is disabled when showpoints is True")
            downsample = 1
        else:
            logger.debug("plot: The shown points don't reflect the actual points"
                         "because downsample > 1")
    return func(spectrum, downsample=downsample, antialias=antialias, exp=exp, 
                linewidth=linewidth, showpoints=showpoints, kind=kind, pitchmode=pitchmode,
                **kws)
    

def plotdraft(spectrum, linewidth=2, exp=1):
    return plot_pyqtgraph(spectrum, linewidth=linewidth, exp=exp, downsample=2, 
                          antialias=False, showpoints=False, numcolors=32)


def plot_pyplot(partials, size=20, alpha=0.5, downsample=1, kind='amp', **kws):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 1)
    if kind == 'amp':
        for p in partials:
            _pyplot_plot_partial_amp(p, ax, size, alpha, downsample=downsample)
    elif kind == 'bw':
        for p in partials:
            _pyplot_plot_partial_bw(p, ax, size, alpha, downsample=downsample)
    plt.show()
    return True


_pyplot_colorcurve_amp = bpf.linear(-90, 0.8, -30, 0.3, -6, 0.1, 0, 0)
_pyplot_colorcurve_bw = bpf.linear(0, 0.15, 0.5, 0.3, 0.9, 0.5, 0.95, 0.8, 1, 0.99)


def _pyplot_plot_partial_amp(partial: _partial.Partial, ax, size=20.0, alpha=0.5,
                             downsample:int=1):
    X = partial.times     # type: np.ndarray
    Y = partial.freqs     # type: np.ndarray
    amps = partial.amps   # type: np.ndarray
    if downsample > 1:
        X = X[::downsample]
        Y = Y[::downsample]
        amps = amps[::downsample]
    dbs = amp2db_np(amps)
    colorcurve = _pyplot_colorcurve_amp
    Z = colorcurve.map(dbs, out=dbs)
    # plt.gray()
    linecol = colorcurve(-70)
    ax.plot(X, Y, color=(linecol, linecol, linecol, 0.5))
    ax.scatter(X, Y, size, c=Z, linewidth=0, alpha=alpha)
    return ax


def _pyplot_plot_partial_bw(partial, ax=None, size=20.0, alpha=0.5, downsample=1):
    X = partial.times     # type: np.ndarray
    Y = partial.freqs     # type: np.ndarray
    Z = partial.amps      # type: np.ndarray
    if downsample > 1:
        X = X[::downsample]
        Y = Y[::downsample]
        Z = Z[::downsample]
    colorcurve = _pyplot_colorcurve_bw
    Z = colorcurve.map(Z)
    # plt.gray()
    linecol = colorcurve(0.01)
    ax.plot(X, Y, color=(linecol, linecol, linecol, 0.5))
    ax.scatter(X, Y, size, c=Z, linewidth=0, alpha=alpha)
    return ax


# noinspection PyUnresolvedReferences
def plot_pyqtgraph(spectrum, *, alpha=1, linewidth=2,
                   exp=1, numcolors=500, downsample=1,
                   antialias=False, showpoints=False,
                   background=None, kind='amp', pitchmode='freq') -> bool:
    try:
        from . import plotpyqtgraph
    except ImportError:
        return False

    if showpoints:
        alpha = 0.6
        exp = exp * 0.8
    plot = plotpyqtgraph.plotspectrum(
        spectrum, alpha=int(alpha*255), linewidth=linewidth, 
        exp=exp, numcolors=numcolors, downsample=downsample,
        antialias=antialias, background=background,
        kind=kind, pitchmode=pitchmode)
    return plot
