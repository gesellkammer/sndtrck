from __future__ import annotations
from bpf4 import bpf
import operator as _operator
import logging
import tempfile
from functools import lru_cache

from . import io
from .config import getconfig
from .util import *
from .partial import Partial
from . import typehints as t


Bpf = bpf.BpfInterface
inf = float("inf")

logger = logging.getLogger("sndtrck")

__all__ = [
    'Spectrum',
    'fromarray',
    'merge',
    'readspectrum',
    'concat',
    'logger'
]


#######################################################################
#
# Spectrum
#
#######################################################################


class Spectrum(object):
    def __init__(self, partials:t.Iter[Partial], *,
                 skipsort=False, packed=False):
        """
        partials: a seq. of Partial (can be a generator).
        To read a saved Spectrum, call Spectrum.read

        
        skipsort: if True, sorting of the partials will be skipped
        packed: metadata to indicate that this Spectrum is the packed
                version of another spectrum, with non-simultaneous
                partials melded into longer tracks, to minimize the
                number of partials (this is done mostly for resynthesis)

        See Also: analyze
        """
        self._t1 = -1.0             
        self._f0 = None            
        self.packed: bool = packed     
        self._sorted: bool = skipsort  
        self.partials = aslist(partials)
        if not skipsort:
            self.partials.sort(key=lambda p:p.t0)
        self.t0: float = partials[0].t0 if partials else 0
        if self.t0 < 0:
            logger.error("Attempting to create a Spectrum negative time: %.2f" % self.t0)
            config = getconfig()
            if not config['spectrum.allow_negative_times']:
                logger.info("Cropping negative part of spectrum")
                partials = [p.crop(0, inf) for p in self.partials]
                self.partials = partials
                self.t0 = 0
                
    def _clear_cache(self):
        self._synthesize.cache_clear()
        
    def _sort(self, force=False):
        if not self._sorted or force:
            partials = self.partials
            if not partials:
                return
            partials.sort(key=lambda p: p.t0)
            self._sorted = True
            self._t0 = partials[0].t0
            self._t1 = -1  # max(p.t1 for p in partials)
        return self

    def __hash__(self):
        # type: () -> int
        s = self.__repr__()
        bps = tuple(p.numbreakpoints for p in self)
        return hash((hash(s), hash(bps)))

    @property
    def duration(self):
        return self.t1 - self.t0
        
    @property
    def t1(self):
        t1 = self._t1
        if t1 < 0:
            self._t1 = t1 = max(p.t1 for p in self.partials)
        return t1

    @classmethod
    def read(cls, path):
        # type: (str) -> Spectrum
        return readspectrum(path)

    def __repr__(self):
        # type: () -> str
        return "Spectrum [%.4f:%.4f]: %d partials" % (
            self.t0, self.t1, len(self.partials))

    def __iter__(self):
        # type: () -> t.Iter[Partial]
        return iter(self.partials)

    def __eq__(self, other: Spectrum) -> bool:
        if not isinstance(other, Spectrum):
            raise TypeError("Can only compare to another Spectrum")
        if self is other:
            return True
        for partial0, partial1 in zip(self.partials, other.partials):
            if partial0 != partial1:
                return False
        return True

    def partials_at(self, t:float) -> Spectrum:
        return self.partials_between(t, t + 1e-9, crop=False)

    def partials_between(self, t0: float, t1: float, crop=False, fade=0.0) -> Spectrum:
        out = []
        if crop:
            t0 += fade
            t1 -= fade
        for partial in self.partials:
            if partial.t0 > t1:
                break
            if partial.t1 >= t0 and partial.t0 <= t1:
                if crop:
                    partial = partial.crop(t0, t1)
                if fade:
                    partial = partial.fade(fade)
                out.append(partial)

        return Spectrum(out, skipsort=True)

    def partials_between_freqs(self, minfreq=0., maxfreq=24000., method="mean") -> Spectrum:
        """
        * method: "weighted"     --> use the weighted meanfreq.
                  "mean"         --> use the meanfreq.
                  "minmax"       --> use the minimum and maximum freqs.
        """
        if method == "weighted":
            out = [p for p in self if minfreq <= p.meanfreq_weighted < maxfreq]
        elif method == "mean":
            out = [p for p in self if minfreq <= p.meanfreq < maxfreq]
        elif method == "minmax":
            out = [p for p in self if minfreq < p.minfreq and maxfreq > p.maxfreq]
        else:
            raise ValueError("partials_between_freqs: method not understood")
        return Spectrum(out, skipsort=True)

    def data_at(self, t:float, maxnotes=0, minamp=-50., mindur=0., minfreq=0., maxfreq=inf
                ) -> t.List[t.Tup[float, float]]:
        """
        A quick way to query a spectrum at a given time

        maxnotes: the max. amount of notes in the chord. 0 for unlimited
        minamp: minimum amplitude to consider
        mindur: consider only partials with a duration greater than this

        Returns a list of tuples (freq, amp), sorted by higher amplitude
        """
        data = []
        minamp = db2amp(minamp)
        partials = self.partials_at(t)
        if minfreq > 0 or maxfreq < inf:
            partials = partials.partials_between_freqs(minfreq, maxfreq)
        for p in partials:
            amp = p.amp(t)
            if amp >= minamp and p.duration > mindur:
                data.append((p.freq(t), amp))
        data.sort(reverse=True, key=_operator.itemgetter(1))
        if maxnotes > 0:
            data = data[:maxnotes]
        return data

    def chord_at(self, t:float, maxnotes=0, minamp=-50., mindur=0., 
                 minfreq=0., maxfreq=inf,
                 ) -> t.List[t.Tup[float, float]]:
        """
        A quick way to query a spectrum at a given time

        maxnotes: the max. amount of notes in the chord. 0 for unlimited
        minamp: minimum amplitude to consider
        mindur: consider only partials with a duration greater than this

        Returns a Chord
        """
        data = self.data_at(t, maxnotes=maxnotes, minamp=minamp, mindur=mindur, minfreq=minfreq,
                            maxfreq=maxfreq)
        notes = [(f2m(freq), amp) for freq, amp in data]
        from emlib.music import core 
        return core.Chord(notes)
        
    def filter(self, mindur=0., minamp=-90., minfreq=0., maxfreq=24000., minbps=1) -> Spectrum:
        """
        Intended for a quick filtering of undesired partials

        Returns a new Spectrum with the partials satisfying 
        ALL the given conditions.

        mindur: min. duration of a partial
        minamp: min. mean amplitude (in dB) of a partial
        minfreq: min. frequency of a partial
        maxfreq: max. frequency of a partial
        minbps: min. number of breakpoints of a partial

        SEE ALSO: filtercurve, adaptive_filter
        """
        out = []
        minamp = db2amp(minamp)
        for p in self.partials:
            if (
                p.duration > mindur and 
                p.meanamp > minamp and 
                (minfreq <= p.meanfreq < maxfreq) and
                p.numbreakpoints >= minbps
            ):
                out.append(p)
        return Spectrum(out, skipsort=self._sorted)

    def equalize(self, curve:Bpf, mindb=-90.) -> Spectrum:
        """
        Equalize all partials in this Spectrum
        
        curve: a bpf mapping freq -> gain 
        mindb: partials below this amplitude are filtered out 
        
        Returns a Spectrum with the equalized partials
        """
        minamp = db2amp(mindb)
        equalized = (p.equalize(curve) for p in self.partials)
        filtered = [p for p in equalized if p.meanamp > minamp]
        return Spectrum(filtered, skipsort=self._sorted)

    def filtercurve(self, freq2minamp:Bpf=None, freq2mindur:Bpf=None) -> t.Tup[Spectrum, Spectrum]:
        """
        return too Spectrums, one which satisfies the given criteria, 
        and the residuum so that both reconstruct the original Spectrum

        freq2minamp: a bpf mapping frequency to min. amplitude in dB
        freq2mindur: a bpf mapping frequency to min. duration

        If a partials is too soft or too short for its frequency, it is
        rejected
        
        Returns
        -------
        
        A tuple(selectedSpectrum, rejectedSpectrum)
        
        SEE ALSO: .filter
        
        Example
        -------

        Filter out weak partials outside the range of musical instruments, 
        preparing for score transcription

        >>> partials = fromtxt("analysis.txt")
        # a low pass frequency curve 
        >>> minampcurve = bpf.linear(0, -90, 1000, -70, 10000, -50, 20000, -30)
        >>> partials.filtercurve(minampcurve)
        # a duration curve filtering short partials 
        >>> freq2mindur = bpf.linear(0, 0, 10000, 0, 15000, 3, 20000, 10)
        >>> partials.filtercurve(freq2mindur=freq2mindur)
        """
        selected, rejected = [], []
        if freq2minamp is not None:
            f2amp = freq2minamp.apply(db2amp)     # type: t.Func1
        else:
            f2amp = lambda x:0.0
        if freq2mindur:
            f2dur = freq2mindur.apply(db2amp)     # type: t.Func1
        else:
            f2dur = lambda x:0.0
        for p in self.partials:
            freq = p.meanfreq_weighted
            minamp = f2amp(freq)
            mindur = f2dur(freq)
            if p.meanamp < minamp or p.duration < mindur:
                rejected.append(p)
            else:
                selected.append(p)
        return (Spectrum(selected, skipsort=True),
                Spectrum(rejected, skipsort=True))

    def timewarp(self, timecurve:Bpf) -> Spectrum:
        partials = [p.timewarp(timecurve) for p in self.partials]
        return Spectrum(partials, skipsort=True)

    def copy(self) -> Spectrum:
        """
        We copy the partials list, since Partials themselves are inmutable
        but the list of them is not
        """
        return Spectrum(self.partials.copy(), skipsort=True)
        
    def __getitem__(self, n):
        if isinstance(n, list):
            ps = self.partials
            return Spectrum([ps[elem] for elem in n], skipsort=True)
        out = self.partials[n]
        if isinstance(out, list):
            return Spectrum(out, skipsort=True)
        else:
            return out
        
    def __len__(self) -> int:
        return len(self.partials)

    def __add__(self, other) -> Spectrum:
        return merge(self, other)

    def gain(self, gain:float) -> Spectrum:
        """
        Example: increase the amplitude of all breakpoints by 6dB
                 (which roughly duplicated the amplitude)

        s2 = s.scaleamp(db2amp(6))
        """
        partials = [p.gain(gain) for p in self.partials]
        return Spectrum(partials, skipsort=True)

    def edit(self) -> Spectrum:
        """ edit the spectrum with an external editor, returns
        the resulting spectrum

        NB: to perform an inplace edit, do
        s = s.edit()
        """
        outfile = self._show_in_spear()
        if not outfile or not os.path.exists(outfile):
            raise RuntimeError("could not reload file: %s" % outfile)
        logger.info("edit: outfile is %s" % outfile)
        import time
        time.sleep(1)
        return readspectrum(outfile)
        
    def plot(self, linewidth:int=None, downsample:int=None, antialias=True, exp=1.0, showpoints=False,
             kind='amp', pitchmode='freq', **kws) -> None:
        """
        Call the default plotting routine. For more customizations, do
        
        from sndtrck import plot
        plot.plot(spectrum, ...)

        Or you can call a specific backend, which exposes its own customization

        downsample: 1 plots all breakpoints, 2 plots 1/2, 3 plots 1/3 
                    of all breakpoints, etc.
        antialias: depending on the backend, can be turned off to
                   have better performance for very large plots
        exp: apply an exponential factor (amps ** exp) over the amplitude 
             to control contrast. 
             * If exp == 0.5, week breakpoints will be more visible
             * If exp == 2, only very strong breakpoints will be visible
        kind: 'amp' or 'bw'
        pitchmode: 'freq' or 'note'
        """
        from .plot import plot
        cfg = getconfig()
        linewidth = linewidth or cfg['plot.linewidth']
        downsample = downsample or cfg['plot.downsample']
        return plot(self, linewidth=linewidth, downsample=downsample, 
                    antialias=antialias, exp=exp, showpoints=showpoints, 
                    kind=kind, pitchmode=pitchmode, **kws)

    def show(self, method:str=None) -> None:
        """
        Show the spectrum. If no method is specified, it uses the
        method in the configuration: getconfig()['spectrum.show.method']

        method: None to use config, one of: "builtin", "spear"
        """
        config = getconfig()
        method = method or config['spectrum.show.method']
        if method == 'builtin':
            from . import interact
            return interact.interact(self)
        elif method == 'spear':
            outfile = self._show_in_spear()
            if outfile and os.path.exists(outfile):
                os.remove(outfile)
        else:
            raise ValueError("Supported options for are 'builtin' or 'spear'")

    def _show_in_spear(self) -> str:
        config = getconfig()
        spearformat = config.get('spearformat', 'sdif')
        if filename is None:
            filename = tempfile.mktemp(suffix = "." + spearformat)
        self.write(filename)
        io.open_spectrum_in_spear(filename, wait=True)
        return filename

    def write(self, outfile:str) -> None:
        """
        Write the spectrum to disk. The format is given by the extension

        txt  : in the format defined by spear
        sdif : SDIF format (RBEP). See .writesdif for more options
        hdf5 / h5 : use a HDF5 based format (needs h5py installed)
        npz  : numpy zipped matrices (cross-platform, always available)
        """
        from . import io
        base, ext = os.path.splitext(outfile)
        matrices = self.asarrays()
        labels = self.labels()
        if ext == '.sdif':
            io.tosdif(matrices, labels, outfile)
        elif ext == '.txt':
            io.tospear(matrices, outfile)
        elif ext == '.npz':
            labels = self.labels()
            io.tonpz(matrices, labels, outfile)
        elif ext == '.hdf5' or ext == '.h5':
            io.tohdf5(matrices, labels, outfile)
        else: 
            raise ValueError("Format not supported")
        
    def writesdif(self, outfile: str, rbep=True, fadetime=0.0) -> None:
        """
        Write this spectrum to sdif

        rbep: saves the data as is, does not resample to 
              a common timegrid. This format is recommended
              over 1TRC if your software supports it
              If False, the partials are resampled to fit to 
              a common timegrid. They are saved in the 1TRC
              format.

        fadetime: if > 0, a fade-in or fade-out will be added
                  (with the given duration) for the partials 
                  which either start or end with a breakpoint
                  with an amplitude higher than 0
        """
        matrices, labels = self.asarrays(), self.labels()
        io.tosdif(matrices, labels, outfile, rbep=rbep, fadetime=fadetime)

    def asarrays(self) -> t.Generator[np.ndarray]:
        """
        Convert each partial to an array, returns a list of such arrays
        """
        return (partial.toarray() for partial in self.partials)
        
    def labels(self) -> t.List[int]:
        """
        Returns the labels (an int) of all partials, as a list
        """
        return [p.label for p in self.partials]

    def resampled(self, dt: float) -> Spectrum:
        """
        Returns a new Spectrum, resampled using dt as sampling period
        """
        partials = [p.resampled(dt) for p in self.partials]
        return Spectrum(partials, skipsort=True)
        
    def fit_between(self, t0: float, t1: float) -> Spectrum:
        """
        Return a new Spectrum, fitted between the given times
        """
        # curve = bpf.linear(self.t0, t0, self.t1, t1)
        # return self.timewarp(curve)
        preoffset = -self.t0
        factor = (t1-t0) / (self.t1-self.t0)
        postoffset = t0
        return self.timescale(factor, preoffset=preoffset, postoffset=postoffset)

    def timescale(self, factor: float, preoffset=0.0, postoffset=0.0) -> Spectrum:
        """
        To map u0, u1 to v0, v1:

        preoffset = -u0
        factor = (v1-v0)/(u1-u0)
        postoffset = v0
        """
        assert factor > 0
        newpartials = []
        for partial in self.partials:
            times = partial.times.copy()  # type: np.ndarray
            if preoffset != 0:
                times += preoffset
            times *= factor
            if postoffset != 0:
                times += postoffset
            partial2 = partial.clone(times=times)
            newpartials.append(partial2)
        return Spectrum(newpartials, skipsort=True)

    def freqwarp(self, curve: t.Func1) -> Spectrum:
        """
        curve: maps freq to freq
        """
        return Spectrum([p.freqwarp(curve) for p in self.partials], skipsort=True)

    def freqwarp_dynamic(self, gradient: t.Fun[[float, float], float]) -> Spectrum:
        # type: (Fun[[float, float], float]) -> Spectrum
        """
        gradient: 
            a callable of the form func(time, freq) -> freq
            Maps freq to freq, varying in time
        """
        return Spectrum([p.freqwarp_dynamic(gradient) for p in self.partials], 
                        skipsort=True)

    def transpose(self, interval: t.U[float, t.Func1]) -> Spectrum:
        """
        interval:
            either a interval as a float, or a function func(t) -> interval
            to perform a time dependent transposition
        """
        partials = [p.transpose(interval) for p in self.partials]
        return Spectrum(partials, skipsort=True)
    
    def shifted(self, dt: t.U[float, t.Func1] = 0, 
                      df: t.U[float, t.Func1] = 0) -> Spectrum:
        """
        Return a new Spectrum shifted in time and/or freq. 

        dt: 
            an amount of time to shift each breakpoint (scalar or func(t) -> dt)
        df: 
            all frequencies are shifted by this ammount (scalar or func(t) -> df)
        """
        partials = [p.shifted(dt=dt, df=df) for p in self.partials]
        return Spectrum(partials, skipsort=True)

    def __lshift__(self, secs):
        return self.shifted(dt=-secs)

    def __rshift__(self, secs):
        return self.shifted(dt=secs)

    @lru_cache(maxsize=10)
    def _synthesize(self, samplerate:int, start:float, end:float, method=None):
        from . import synthesis
        return synthesis.render_samples(self, samplerate, start=start, end=end, 
                                        method=method)
        
    def synthesize(self, sr:int=44100, start:float=-1, end:float=-1
                   ) -> np.ndarray:
        """
        Synthesize this Spectrum

        * sr: the samplerate of the synthesized samples
        
        See .render to render this spectrum to disk
        """
        if start < 0: 
            start = self.t0
        if end < 0: 
            end = self.t1
        return self._synthesize(sr, start=start, end=end)

    def render(self, sndfile:str, sr:int=44100, start:float=-1, end:float=-1) -> None:
        """
        Render this spectrum as a soundfile to disk

        This is in theory the same as spectrm.synthesize(sr).write("sndfile.wav"),
        but it is recommended since the backend can then decide if it is more
        efficient to generate the samples directly to disk.

        To configure the synthesis method, see getconfig()['render.method']
        """
        from . import synthesis
        return synthesis.render_sndfile(self, sr=sr, outfile=sndfile, start=start, end=end)

    def play(self, start=-1., end=0., loop=False, speed=1., play=True, gain=1.
            ) -> 'synthesis.SpectrumPlayer':
        """
        Play the spectrum in real-time. Returns a SpectrumPlayer, 
        which can be controlled while sound is playing

        start: start time. A negative number will start at the beginning of the spectrum
               (skipping any silence before t=0)
        end: end time. A negative will count time from the end
        loop: if True, loop between start and end
        speed: Playback speed. NB: 0 freezes the sound.
        play: if True, start playing right away.
        """
        from . import synthesis
        if start < 0:
            start = self.t0
        if end <= 0:
            end = self.t1 - end
        verbose = logger.level == logging.DEBUG
        player = synthesis.SpectrumPlayer(self, autoplay=play, looping=loop,
                                          speed=speed, start=start, end=end,
                                          exitwhendone=(not loop), interactive=True,
                                          gain=gain, verbose=verbose)
        return player
        
    def estimatef0(self, minfreq:float, maxfreq:float, interval=.1) -> io.EstimateF0:
        """
        Estimate the fundamental of this spectrum

        * minfreq, maxfreq: the frequency range where a fundamental is expected
        * interval: the time accuracy of the measurement

        Returns: namedtuple(freq, confidence), where:

            - freq: a bpf representing the frequency of the fundamental at time `t`
            - confidence: a bpf representing the confidence of the measurement at time `t`
                    Values of 0.9-1 represent high confidence, 0 represents no partial
                    at time `t`, values in between represent unvoiced sounds 
        """
        if self._f0 is not None:
            return self._f0
        matrices = self.asarrays()
        self._f0 = io.estimatef0(matrices, minfreq, maxfreq, interval)
        return self._f0

    def normalize(self, maxpeak):
        # type: (float) -> Spectrum
        """
        Normalize all partials based on the maximum amp. value of
        this spectrum. 

        Example: normalize a spectrum so that it does not clip

        samples = s.synthesize(s2, 22000)
        maxpeak = np.abs(samples).max()
        s2 = s.normalize
        """
        raise NotImplementedError

    def simplify(self, freqdelta=-1, ampdelta=-1, bwdelta=-1):
        # type: (float, float, float) -> Spectrum
        """ 
        Simplify each partial 

        pitchdelta: min. difference in semitones
        dbdetal: min. difference in dBs
        bwdelta: min. difference in bandwidth
        """
        newpartials = [p.simplify(freqdelta=freqdelta, ampdelta=ampdelta, bwdelta=bwdelta)
                       for p in self]
        return Spectrum(newpartials, skipsort=True)

    def estimate_breakpoint_gap(self, percentile=None, partial_percentile=None):
        # type: (float, float) -> float
        """
        Estimate the breakpoint gap in this spectrum. 
        
        To configure the defaults, see keys: 'breakpointgap.percentile' and 
        'breakpointgap.partial_percentile'
        """
        cfg = getconfig()
        percentile = percentile if percentile is not None else cfg['breakpointgap.percentile']
        partial_percentile = cfg.override(partial_percentile, 'breakpointgap.partial_percentile')
        return _estimate_gap(self, percentile=percentile, partial_percentile=partial_percentile)
        

def _default(value, key, config=None):
    config = config or getconfig()
    if value is not None:
        return value
    return config[key]


@lru_cache(maxsize=128)
def _estimate_gap(sp:Spectrum, percentile:int, partial_percentile:int):
    values = [p.estimate_breakpoint_gap(partial_percentile) for p in sp.partials
              if p.numbreakpoints > 2]
    value = np.percentile(values, percentile)
    return value


# <--------------------------------- END Spectrum


def fromarray(arrayseq, labels=None):
    # type: (t.List[np.ndarray], t.Opt[t.List[int]]) -> Spectrum
    """
    construct a Spectrum from array data

    arrayseq: a seq. of matrices, where each matrix is a 2D array 
              with columns -> [time, freq, amp, phase, bw]
    """
    newpartials = []
    if labels is None:
        labels = [0] * len(arrayseq)
    minbreakpoints = getconfig()['spectrum.minbreakpoints']
    for matrix, label in zip(arrayseq, labels):
        if len(matrix) < minbreakpoints:
            logger.debug(f"fromarray: Skipping short partial of #bps={len(matrix)}")
            continue
        newpartials.append(Partial.fromarray(matrix, label=label))
    return Spectrum(newpartials)


def merge(*spectra):
    # type: (*Spectrum) -> Spectrum
    """
    Merge two spectra into a third one
    """
    partials = []
    for spectrum in spectra:
        partials.extend(spectrum.partials)
    s = Spectrum(partials)
    return s


def concat(spectra, separator=0, strip=True):
    # type: (t.List[Spectrum], float, bool) -> Spectrum
    """
    Concatenate (juxtapose) multiple spectra

    separator: a number indicated the duration of a silence, or 
               a Spectrum to intercalate between the other spectra
    strip: if True, it removes any silence at the beginning of a Spectrum
           before concatenation
    """
    def addspectrum(partials, spectrum, t, strip, separator):
        dt = t
        if strip and spectrum.t0 > 0:
            dt = dt - spectrum.t0
        spectrum = spectrum.shifted(dt)
        partials.extend(spectrum.partials)
        t = spectrum.t1
        if isinstance(separator, (int, float)):
            t += separator
        return t
    partials = []  # type: t.List[Partial]
    now = 0
    for spectrum in spectra[:-1]:
        now = addspectrum(partials, spectrum, now, strip, separator)
    # now add the last one
    addspectrum(partials, spectra[-1], now, strip, separator=0)
    return Spectrum(partials)
 

#################################################
# IO
#################################################


def _namedtup2partial(p):
    """
    p is a namedtuple _Partial
    """
    return Partial(p.time, p.freq, p.amp, phases=p.phase, bws=p.bw, label=p.label)


def readspectrum(path):
    """
    Read a spectrum, returns a Spectrum

    Supported formats: .txt, .sdif, .h5, .hdf5, .npz
    """
    funcs = {
        '.txt': io.fromtxt,
        '.hdf5': io.fromhdf5,
        '.h5': io.fromhdf5,
        '.sdif': io.fromsdif,
        '.npz': io.fromnpz
    }
    ext = os.path.splitext(path)[1]
    if ext not in funcs:
        raise ValueError("Format not supported")
    matrices, labels = funcs[ext](path)
    return fromarray(matrices, labels)


def harmonicdev(f0, partial):
    """
    How much does partial deviate from being a harmonic of f0?

    Returns: a value between 0 and 0.5

    TODO: take correlation into account
    """
    logger.warn("Depcreted. Use Partial.harmonicdev")
    return partial.harmonicdev(f0)


