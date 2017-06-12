"""
loris backend based on loristrck

Will only work if loristrck is available. Loristrck is a simple wrapper 
around the partial tracking library Loris, which differs from its built-in 
python bindings in that it is not swig generated but implemented in cython. 
It does not need Loris itself to be installed: it links directly to it and
is compiled on the package itself. This makes installation much easier and
reliable, since there are no PATH problems at compile or at runtime.
Loris is only used to analyze the sound and is converted to an agnostic 
data representation based on numpy arrays. This makes it easier to manipulate
(the Loris bindings are not very nice to use from a python stand-point)

A backend should implement:

is_available() -> bool
get_info() --> dict 
               {
                'analyze': bool,
                'read_sdif': bool,
                'write_sdif': bool
               }

"""
from __future__ import absolute_import
import warnings as _warnings
import numpy as np
from .errors import BackendNotAvailable
from .util import isiterable, sndreadmono, aslist
import bpf4 as bpf
from typing import Iterable as Iter, Optional as Opt, List, Tuple as Tup, Union
from .log import get_logger
import logging

logger = get_logger()

try:
    import loristrck
    AVAILABLE = True
    logger.info("backend loristrck OK")
    loristrck.logger.setLevel(logging.INFO)
except ImportError:
    _warnings.warn("loristrck is not available, the loris backend cannot be used")
    logger.info("backend loristrck not found!")
    AVAILABLE = False
    loristrck = None

# A backend should implement these functionality:
#    * is_available() -> bool
#    * get_info()     -> dict with keys defining the available functionality

    
def is_available():
    # type: () -> bool
    return AVAILABLE


def get_info():
    # type: () -> dict
    return {
        'name': 'loris',
        'analyze': True,
        'read_sdif': True,
        'write_sdif': True,
        'synthesize': True,
        'estimatef0': True,
        'analysis_options': {
            'freqdrift': 'max. drift in freq. between 2 breakpoints',
            'hop': 'hop time as a fraction of window (0-1)',
            'sidelobe': 'shape of the Kaiser window (in +dB). Default=-ampfloor',
            'ampfloor': 'only breakpoints above this amplitude are kept',
            'croptime': 'max. time correction between 2 breakpoints. Default=hoptime'
        }
    }


def _default(val, default):
    return val if val is not None else default


def analyze_samples(samples,    # type: np.ndarray
                    sr,         # type: int
                    resolution, # type: float
                    windowsize=None,  # type: Opt[float]
                    hop=None,         # type: Opt[float]
                    freqdrift=None,   # type: Opt[float]
                    sidelobe=None,    # type: Opt[float]
                    ampfloor=None,    # type: Opt[float]
                    croptime=None     # type: Opt[float]
    ):
    # type: (...) -> List[np.ndarray]
    """
    Partial tracking analysis
    
    :param samples: the samples to analyze
    :param sr: sampling rate
    :param resolution: resolution of the analysis in Hz
    :param windowsize: size of the analysis window **in Hz**
    :param hop: hoptime, as a fraction of a window (sensible values: 1/8 to 1)
    :param freqdrift: max. drift in freq between 2 breakpoints (a normal default is 1/2 of the resolution)
    :param sidelobe: shape of the kaiser window (in +dB)
    :param ampfloor: only breakpoints above this amplitude are kept for partial tracking
    :param croptime: max. time correction between 2 breakpoints
    :return: a list of numpy arrays, where each array represents a partial
             The shape of each array is (numbreakpoints, 5), with columns: time, freq, amp, phase and bw
    """
    if not AVAILABLE:
        raise BackendNotAvailable("loristrck not available")

    freqdrift = _default(freqdrift, 0.75*resolution)
    ampfloor = _default(ampfloor, -90)
    sidelobe = _default(sidelobe, -ampfloor)    # we leave -1 to let the backend set its default (-ampfloor)
    windowsize = _default(windowsize, 2*resolution)
    croptime = _default(croptime, -1)
    hop = _default(hop, 1)
    hoptime = (1.0 / windowsize) * hop  # originl Loris behaviour is hop=1
    logger.info("analyze_samples: using windowsize = {}".format(windowsize))
    partialdata = loristrck.analyze(samples, sr, resolution, windowsize,
                                    hoptime=hoptime, freqdrift=freqdrift, sidelobe=sidelobe,
                                    ampfloor=ampfloor, croptime=croptime)
    return partialdata


def analyze(sndfile,          # type: Union[str, Tup[np.ndarray, int]]
            resolution,       # type: float
            windowsize=None,  # type: Opt[float]
            hop=None,         # type: Opt[float]
            freqdrift=None,   # type: Opt[float]
            sidelobe=None,    # type: Opt[float]
            ampfloor=None,    # type: Opt[float]
            croptime=None,    # type: Opt[float]
            channel=1         # type: int
    ):
    # type: (...) -> List[np.ndarray]
    """
    Partial tracking analysis
    
    :param sndfile: the path to a soundfile, or a tuple(samples, samplerate) 
    :param resolution: the resolution of the analysis, in Hz
    :param windowsize: size of the analysis window **in Hz**
    :param hop: hoptime, as a fraction of a window (sensible values: 1/8 to 1)
    :param freqdrift: max. drift in freq between 2 breakpoints (a normal default is 1/2 of the resolution)
    :param sidelobe: shape of the kaiser window (in +dB)
    :param ampfloor: only breakpoints above this amplitude are kept for partial tracking
    :param croptime: max. time correction between 2 breakpoints
    :param channel: the channel to use for analysis if given a stereo file. 
    :return: a list of numpy arrays, where each array represents a partial
             The shape of each array is (numbreakpoints, 5), with columns: time, freq, amp, phase and bw
    """
    if isinstance(sndfile, str):
        samples, sr = sndreadmono(sndfile, channel)
    elif isinstance(sndfile, tuple):
        samples, sr = sndfile
    else:
        raise TypeError("sndfile should be either a string path to a soundfile or a tuple(samples, sr)")
    samples = np.ascontiguousarray(samples)
    return analyze_samples(samples, sr, resolution, windowsize, hop,
                           freqdrift=freqdrift, sidelobe=sidelobe, ampfloor=ampfloor,
                           croptime=croptime)


def synthesize(matrices, samplerate=44100, fadetime=None):
    # type: (Iter[np.ndarray], int, Opt[float]) -> np.ndarray[float]
    """
    Returns a numpy 1D array holding the samples 
    
    matrices: a seq. of 2D matrices, where each matrix represents a partial

    Example:

    matrices = analyze("path/to/sound", ...)
    samples = synthesize(matrices, 44100)
    sndfileio.sndwrite(samples, 44100, "/path/to/newsound.wav")
    """
    assert isiterable(matrices)
    assert isinstance(samplerate, int)
    if not AVAILABLE:
        raise BackendNotAvailable("loristrck not available")
    if fadetime is None:
        # 32 samples ramp
        fadetime = 32./samplerate
    samples = loristrck.synthesize(matrices, samplerate, fadetime)
    return samples


def read_sdif(sdiffile):
    # type: (str) -> Tup[List[np.ndarray], List[int]]
    """
    Reads partial-tracking sdiffile, returns a tuple (partials, labels).
    Formats supported: 1TRK, RBEP

    * partials: a list of 2D numpy.arrays, where each array represents a partial.
                Each row of the array represents a breakpoint with columns
                [time, freq, amplitude, phase, bandwidth]
    * labels: a list of numeric labels (if not present, this will be a list of 0s)

    To construct a Spectrum, call fromarray(read_sdif(sdiffile))
    """
    if not AVAILABLE:
        raise BackendNotAvailable("loristrck not available")
    partials, labels = loristrck.read_sdif(sdiffile)
    return partials, labels


def write_sdif(matrices, outfile, labels=None, rbep=True, fadetime=0):
    # type: (Iter[np.ndarray], str, Opt[Iter[int]], bool, float) -> None
    """
    matrices: a seq. of 2D numpy arrays where each matrix represents a partial.
              Partial: rows represent breakpoints, each breakpoint has the form
              [time, freq, amp, phase, bw]
    labels: if given, a list of numeric labels, one for each partial (len(matrices) == len(labels))
    rbep: if True, use the RBEP format. Otherwise, the 1TRC format is used
    fadetime: fade the partials to 0 if they end in non-0 amplitude, to avoid clicks when
              synthesizing (a fadetime of 0 disables fading)
    """
    assert isinstance(outfile, str)
    assert isiterable(matrices)
    assert isiterable(labels) or labels is None
    assert fadetime >= 0
    if not AVAILABLE:
        raise BackendNotAvailable("loristrck not available")
    loristrck.write_sdif(matrices, outfile, labels=labels, rbep=rbep, fadetime=fadetime)


def estimatef0(matrices, minfreq=30, maxfreq=3000, interval=0.1):
    # type: (Iter[np.ndarray], int, int, float) -> Tup[bpf.BpfInterface, bpf.BpfInterface]
    """
    Estimate the funamental freq. of the analysed partials
    
    :param matrices: a seq. of 2D numpy arrays where each matrix represents a partial, as
                     returned by `analyze` 
    :param minfreq: the min. frequency to look for f0
    :param maxfreq: the max. frequency to look to f0
    :param interval: the time interval at which the f0 is estimated
    :return: a tuple(freq, conf), both a bpf. freq represents the freq of the f0 at time t,
             conf is a curve representing the confidence of the measurement at time t
    """
    freqs, confs, t0, t1 = loristrck.estimatef0(matrices, minfreq, maxfreq, interval)
    freqbpf = bpf.core.Sampled(freqs, interval, t0)
    confbpf = bpf.core.Sampled(confs, interval, t0)
    return freqbpf, confbpf
