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
import numpy as np
from .errors import BackendNotAvailable
from .util import isiterable, sndreadmono
import bpf4 as bpf
from typing import Iterable as Iter, Optional as Opt, List, Tuple as Tup, Union as _U
import logging

logger = logging.getLogger("sndtrck")

try:
    import loristrck
    AVAILABLE = True
    logger.debug("backend loristrck OK")
except ImportError:
    logger.error("backend loristrck not found!")
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


def analyze_samples(samples,             # type: np.ndarray
                    sr,                  # type: int
                    resolution,          # type: float
                    windowsize=None,     # type: Opt[float]
                    hop=None,            # type: Opt[float]
                    freqdrift=None,      # type: Opt[float]
                    sidelobe=None,       # type: Opt[float]
                    ampfloor=None,       # type: Opt[float]
                    croptime=None,       # type: Opt[float]
                    residuebw=None,      # type: Opt[float]
                    convergencebw=None,  # type: Opt[float]
    ):
    # type: (...) -> List[np.ndarray]
    """
    For documentation, see analyze
    """
    if not AVAILABLE:
        raise BackendNotAvailable("loristrck not available")

    freqdrift = _default(freqdrift, 0.75*resolution)
    ampfloor = _default(ampfloor, -90)
    sidelobe = _default(sidelobe, -ampfloor)    # we leave -1 to let the backend set its default (-ampfloor)
    windowsize = _default(windowsize, 2*resolution)
    croptime = _default(croptime, -1)
    hop = _default(hop, 1)
    residuebw = _default(residuebw, -1)
    convergencebw = _default(convergencebw, -1)
    hoptime = (1.0 / windowsize) * hop  # originl Loris behaviour is hop=1
    logger.debug("analyze_samples: using windowsize = {}".format(windowsize))
    partialdata = loristrck.analyze(samples, sr, resolution, windowsize,
                                    hoptime=hoptime, freqdrift=freqdrift, sidelobe=sidelobe,
                                    ampfloor=ampfloor, croptime=croptime,
                                    residuebw=residuebw, convergencebw=convergencebw)
    return partialdata


def analyze(sound,               # type: _U[str, Tup[np.ndarray, int]]
            resolution,          # type: float
            windowsize=None,     # type: Opt[float]
            hop=None,            # type: Opt[float]
            freqdrift=None,      # type: Opt[float]
            sidelobe=None,       # type: Opt[float]
            ampfloor=None,       # type: Opt[float]
            croptime=None,       # type: Opt[float]
            residuebw=None,      # type: Opt[float]
            convergencebw=None,  # type: Opt[float]
            channel=0,           # type: int
            start=0.0,
            end=0.0):   # type: (...) -> List[np.ndarray]
    """
    Partial tracking analysis
    
    sound: str or (samples, samplerate)
        the path to a soundfile, or a tuple(samples, samplerate) 
    resolution: Hz
        The resolution of the analysis, in Hz
        Only one partial will be found within this distance. Usual values range 
        from 30 Hz to 200 Hz. As a rule of thumb, when tracking a monophonic source, 
        resolution ~= min(f0) * 0.9 
    windowsize: Hz
        size of the analysis window. If not given, windowsize=2*resolution. It should not
        excede this value, but it can be smaller. The windowsize is inverse to its length
        in samples (see loristrck.kaiserWindowLength) 
    hop: 
        hoptime, as a fraction of a the windowsize. The normal hoptime is 1/windowsize,
        so a hop time of 2 will double that. Most of the times this should be left
        untouched (sensible values: 1/4 to 4). At its default, this results in an
        overlap of ~8x
    freqdrift:  
        max. drift in freq between 2 breakpoints (default is 1/2 of the resolution)
        A sensible value is between 1/2 to 3/4 of resolution
    sidelobe: dB
        shape of the kaiser window (in +dB)
    ampfloor: dB
        only breakpoints above this amplitude are kept for partial tracking
    croptime: sec
        max. time correction between 2 breakpoints (defaults to hop time)
    channel: int
        the channel to use for analysis if given a stereo file.
    residuebw: Hz (default = 2000 Hz)
        Bandwidth env. is created by associating residual energy to 
        the peaks. The value indicates the width of association regions used.
        Defaults to 2 kHz, corresponding to 1 kHz region center spacing.
        NB: if residuebw is set, convergencebw must be left unset
    convergencebw: range [0, 1]
        Bandwidth env. is created by storing the mixed derivative of short-time 
        phase, scaled and shifted. The value is the amount of range over which the 
        mixed derivative is allowed to drift away from a pure sinusoid 
        before saturating. This range is mapped to bandwidth values on 
        the range [0,1].  
        NB: one can set residuebw or convergencebw, but not both 
    start, end: 
        Read a portion of the soundfile. For end, use negative times to count
        from the end
    
    Returns: a list of numpy arrays, where each array represents a partial
             The shape of each array is (numbreakpoints, 5), with columns: 
             time, freq, amp, phase, bw
    """
    if isinstance(sound, str):
        samples, sr = sndreadmono(sound, channel, start=start, end=end)
    elif isinstance(sound, tuple):
        samples, sr = sound
        assert isinstance(samples, np.ndarray)
        if len(samples.shape) > 1:
            logger.error("Multichannel sound found, using channel 0")
            samples = samples[:,0]
    else:
        raise TypeError("sound should be a path or a tuple(samples, sr)")
    if not samples.flags.contiguous:
        samples = np.ascontiguousarray(samples)
    return analyze_samples(samples, sr=sr, resolution=resolution, windowsize=windowsize, hop=hop,
                           freqdrift=freqdrift, sidelobe=sidelobe, ampfloor=ampfloor,
                           croptime=croptime, residuebw=residuebw, convergencebw=convergencebw)


def synthesize(matrices: Iter[np.ndarray],
               samplerate=44100,
               fadetime: float=None,
               start: float=0,
               end: float=0) -> np.ndarray:
    """
    Returns a numpy 1D array holding the samples 
    
    matrices: a seq. of 2D matrices, where each matrix represents a partial
    samplerate: the samplerate of the synthesized sampled
    fadetime: apply this fadetime to each partial to avoid clicks
    start, end: start and end time of synthesis. Leave both in 0 to synthesize all
    
    
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
    samples = loristrck.synthesize(matrices, samplerate, fadetime, start=start, end=end)
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


def write_sdif(outfile, matrices, labels=None, rbep=True, fadetime=0):
    # type: (str, Iter[np.ndarray], Opt[Iter[int]], bool, float) -> None
    """
    matrices: 
        a seq. of 2D numpy arrays where each matrix represents a partial.
        Partial: rows represent breakpoints, each breakpoint has the form
        [time, freq, amp, phase, bw]
    labels: 
        if given, a list of numeric labels, one for each partial 
        NB: len(matrices) == len(labels)
    rbep: 
        if True, use the RBEP format. Otherwise, the 1TRC format is used
    fadetime: 
        fade the partials to 0 if they end in non-0 amplitude, to avoid clicks when
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
    
    matrices: 
        a seq. of 2D numpy arrays where each matrix represents a partial, as
        returned by `analyze` 
    minfreq: 
        the min. frequency to look for f0
    maxfreq: 
        the max. frequency to look to f0
    interval: 
        the time interval at which the f0 is estimated
    
    Returns:
        a tuple(freq, conf), both a bpf. 
        freq represents the freq of the f0 at time t,
        conf is a curve representing the confidence of the measurement at time t
    """
    freqs, confs, t0, t1 = loristrck.estimatef0(matrices, minfreq, maxfreq, interval)
    freqbpf = bpf.core.Sampled(freqs, interval, t0)
    confbpf = bpf.core.Sampled(confs, interval, t0)
    return freqbpf, confbpf
