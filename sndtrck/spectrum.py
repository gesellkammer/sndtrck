from __future__ import division as _division
from __future__ import absolute_import
from bpf4 import bpf
import operator as _operator
import tempfile
from emlib.pitch import db2amp
from emlib.iterlib import pairwise

from . import music
from . import io
from . import log
from .config import CONFIG
from .util import *
from .partial import Partial
from typing import Iterable as Iter, List, Callable, Optional as Opt, Tuple, Iterator

Bpf = bpf.BpfInterface
inf = float("inf")

logger = log.get_logger()


#######################################################################
#
# Spectrum
#
#######################################################################


class Spectrum(object):
    def __init__(self, partials):
        # type: (Iter[Partial]) -> None
        """
        partials: a seq. of Partial (can be a generator).
        To read a saved Specrum, call Spectrum.read

        See Also: analyze
        """
        self.partials = []            # type: List[Partial]
        self._last_written_file=""  # type: str
        self._f0 = None  # type: Opt[io.EstimateF0]
        self.t0 = 0.0    # type: float
        self.t1 = 0.0    # type: float
        self._set_partials(partials)
        if self.t0 < 0:
            logger.warning("Attempted to create a Spectrum with a negative time: %.2f" % self.t0)
            if not CONFIG['allow_negative_times']:
                shifted = self.shifted(dt= -self.t0)
                self._set_partials(shifted.partials)
                logger.warning("Shifed Spectrum to time=%.2f" % self.t0)

    def _set_partials(self, partials):
        # type: (Iter[Partial]) -> None
        self.partials = list(partials)
        assert all(isinstance(partial, Partial) for partial in self.partials)
        self._reset()

    def _reset(self):
        self._sort_if_needed()
        self.t0 = min(p.t0 for p in self.partials) if self.partials else 0
        self.t1 = max(p.t1 for p in self.partials) if self.partials else 0
        self._f0 = -1

    def _sort_if_needed(self):
        if any(p0.t0 > p1.t0 for p0, p1 in pairwise(self.partials)):
            self.partials.sort(key=lambda p: p.t0)

    @classmethod
    def read(cls, path):
        # type: (str) -> Spectrum
        return readspectrum(path)

    def __repr__(self):
        # type: () -> str
        return "Spectrum [%.4f:%.4f]: %d partials" % (
            self.t0, self.t1, len(self.partials))

    def __iter__(self):
        # type: () -> Iterator[Partial]
        return iter(self.partials)

    def __eq__(self, other):
        # type: (Spectrum) -> bool
        if not isinstance(other, Spectrum):
            raise TypeError("Can only compare to another Spectrum")
        for partial0, partial1 in zip(self.partials, other.partials):
            if partial0 != partial1:
                return False
        return True

    def partials_at(self, t):
        # type: (float) -> Spectrum
        return self.partials_between(t, t + 1e-9, crop=False)

    def partials_between(self, t0, t1, crop=False):
        # type: (float, float, bool) -> Spectrum
        out = []
        for partial in self.partials:
            if partial.t1 >= t0 and partial.t0 <= t1:
                if crop and (partial.t1 > t1 or partial.t0 < t0):
                    partial = partial.crop(t0, t1)
                out.append(partial)
            if partial.t0 > t1:
                break
        return Spectrum(out)

    def partials_between_freqs(self, minfreq=0., maxfreq=24000., method="mean"):
        # type: (float, float, str) -> Spectrum
        """
        * method: "weightedmean" --> use the weighted meanfreq.
                  "mean"         --> use the meanfreq.
        """
        if method == "weightedmean":
            out = [p for p in self if minfreq <= p.meanfreq_weighted < maxfreq]
        elif method == "mean":
            out = [p for p in self if minfreq <= p.meanfreq < maxfreq]
        else:
            raise ValueError("partials_between_freqs: method not understood")
        return Spectrum(out)

    # def harmonics(self, partial, maxharmonics=5, threshold=0.2):
    #     assert isinstance(partial, Partial)
    #     f0 = partial.meanfreq
    #     possiblefreqs = [f0*i for i in range(maxharmonics)]
    #     out = []
    #     for candidate in self.partials_between(partial.t0, partial.t1):
    #         if candidate.meanfreq < f0*1.5:
    #             continue

    #         t0, t1 = max(partial.t0, candidate.t0), min(partial.t1, candidate.t1)
    #         assert t1 > t0
    #         freqs0 = partial.freq[t0:t1:0.1]
    #         freqs1 = candiate.freq[t0:t1:0.1]
    #         prod = freqs1/freqs0
    #         dev = numpy.abs(prod - prod.round()).mean()
    #         if dev < threshold:
    #             out.append(candidate)
    #     return out

    def chord_at(self, t, maxnotes=0, minamp=-50, mindur=0, minfreq=0, maxfreq=inf):
        # type: (float, int, float, float, float, float) -> music.Chord
        """
        A quick way to query a spectrum at a given time

        maxnotes: the max. amount of notes in the chord. 0 for unlimited
        minamp: minimum amplitude to consider
        mindur: consider only partials with a duration greater than this
        """
        data = []
        minamp = db2amp(minamp)
        partials = self.partials_at(t)
        if minfreq > 0 or maxfreq < inf:
            partials = partials.partials_between_freqs(minfreq, maxfreq)
        for p in partials:
            amp = p.amp(t)
            if amp >= minamp and p.duration > mindur:
                data.append((f2m(p.freq(t)), amp))
        data.sort(reverse=True, key=_operator.itemgetter(1))
        if maxnotes > 0:
            data = data[:maxnotes]
        out = music.newchord(data)
        assert isinstance(out, music.Chord)
        return out

    def filter(self, mindur=0, minamp=-90, minfreq=0, maxfreq=24000):
        # type: (float, float, float, float) -> Spectrum
        """
        Intended for a quick filtering of undesired partials

        Returns a new Spectrum with the partials satisfying 
        ALL the given conditions.

        SEE ALSO: filtercurve
        """
        out = []
        minamp = db2amp(minamp)
        for p in self.partials:
            if (
                p.duration > mindur and 
                p.meanamp > minamp and 
                (minfreq <= p.meanfreq < maxfreq)
            ):
                out.append(p)
        return Spectrum(out)

    def equalize(self, curve, mindb=-90):
        # type: (Bpf, float) -> Spectrum
        """
        Equalize all partials in this Spectrum
        
        :param curve: a bpf mapping freq -> gain 
        :param mindb: partials below this amplitude are filtered out 
        :return: a Spectrum with the equalized partials
        """
        minamp = db2amp(mindb)
        equalized = (p.equalize(curve) for p in self.partials)
        filtered = [p for p in equalized if p.meanamp > minamp]
        return Spectrum(filtered)

    def filtercurve(self, freq2minamp=None, freq2mindur=None):
        # type: (Opt[Bpf], Opt[Bpf]) -> Tuple[Spectrum, Spectrum]
        """
        return too Spectrums, one which satisfies the given criteria, 
        and the resisuum so that both reconstruct the original Spectrum

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
            f2amp = freq2minamp.apply(db2amp)     # type: Callable[[float], float]
        else:
            f2amp = lambda x:0.0
        if freq2mindur:
            f2dur = freq2mindur.apply(db2amp) # type: Callable[[float], float]
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
        return Spectrum(selected), Spectrum(rejected)

    def timewarp(self, timecurve):
        partials = [p.timewarp(timecurve) for p in self.partials]
        return self.__class__(partials)

    def copy(self):
        return self.__class__(self.partials[:])

    def __getitem__(self, n):
        # TODO: support slicing
        return self.partials[n]

    def __len__(self):
        return len(self.partials)

    def __add__(self, other):
        # type: (Spectrum) -> Spectrum
        partials = self.partials + other.partials
        partials.sort(key=lambda p:p.t0)
        return Spectrum(partials)

    def scaleamp(self, gain):
        # type: (float) -> Spectrum
        partials = [p.scaleamp(gain) for p in self.partials]
        return Spectrum(partials)

    def show(self):
        showapp = CONFIG.get('showprogram', 'spear')
        if showapp == 'spear':
            outfile = self._show_in_spear()
            if outfile and os.path.exists(outfile):
                os.remove(outfile)
        else:
            raise ValueError("Supported options for showprogram are 'spear'")

    def edit(self):
        """ edit the spectrum with an external editor, returns
        the resulting spectrum

        NB: to perform an inplace edit, do
        s = s.edit()
        """
        outfile = ".edit.sdif"
        self._show_in_spear(outfile)
        if not outfile or not os.path.exists(outfile):
            raise RuntimeError("could not reload file: %s" % outfile)
        logger.info("edit: outfile is %s" % outfile)
        import time
        time.sleep(1)
        out = readspectrum(outfile)
        # os.remove(outfile)
        return out
        
    def plot(self, linewidth=2, downsample=1, antialias=True, exp=1, showpoints=False,
             **kws):
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
        """
        from . import plot
        plot.plot(self, linewidth=linewidth, downsample=downsample, 
                  antialias=antialias, exp=exp, showpoints=showpoints, 
                  **kws)

    def _show_in_spear(self, filename=None):
        spearformat = CONFIG.get('spearformat', 'sdif')
        if filename is None:
            filename = tempfile.mktemp(suffix = "." + spearformat)
        self.write(filename)
        io.open_spectrum_in_spear(filename, wait=True)
        return filename

    def write(self, outfile):
        # type: (str) -> None
        """
        Write the spectrum to disk. The format is given by the extension

        txt  : in the format defined by spear
        sdif : SDIF format (RBEP). See .writesdif for more options
        hdf5 / h5 : use a HDF5 based format (needs h5py installed)
        npz  : numpy zipped matrices (cross-platform, always available)
        """
        from . import io
        funcs = {
            '.txt': io.tospear,
            '.hdf5': io.tohdf5,
            '.h5': io.tohdf5,
            '.sdif': io.tosdif,
            '.npz': io.tonpz
        }
        base, ext = os.path.splitext(outfile)
        self._last_written_file = outfile
        if ext not in funcs:
            raise ValueError("Format not supported")
        matrices, labels = self.toarray()
        return funcs[ext](matrices, labels, outfile)

    def writesdif(self, outfile, rbep=True, fadetime=0.0):
        # type: (str, bool, float) -> None
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
        matrices, labels = self.toarray()
        io.tosdif(matrices, labels, outfile, rbep=rbep, fadetime=fadetime)

    def toarray(self):
        # type: () -> Tuple[Iter[np.ndarray], List[int]]
        """
        Returns a tuple (matrices, labels), where matrices is an list of
        matrix --> 2D array with columns [time freq amp phase bw]
        """
        matrices = (partial.toarray() for partial in self.partials)
        labels = [partial.label for partial in self.partials]
        return matrices, labels

    def resampled(self, dt):
        # type: (float) -> Spectrum
        """
        Returns a new Spectrum, resampled using dt as sampling period
        """
        partials = [p.resampled(dt) for p in self.partials]
        return Spectrum(partials)

    def fit_between(self, t0, t1):
        # type: (float, float) -> Spectrum
        """
        Return a new Spectrum, fitted between the given times
        """
        # curve = bpf.linear(self.t0, t0, self.t1, t1)
        # return self.timewarp(curve)
        preoffset = -self.t0
        factor = (t1-t0) / (self.t1-self.t0)
        postoffset = t0
        return self.timescale(factor, preoffset=preoffset, postoffset=postoffset)
        
    def timescale(self, factor, preoffset=0.0, postoffset=0.0):
        # type: (float, float, float) -> Spectrum
        """
        To map u0, u1 to v0, v1:

        preoffset = -u0
        factor = (v1-v0)/(u1-u0)
        postoffset = v0

        The same can be accomplished by:

        (s >> preoffset).timescale(factor) >> postoffset
        """
        assert factor > 0
        newpartials = []
        for partial in self.partials:
            times = partial.times  # type: np.ndarray
            times += preoffset
            times *= factor
            times += postoffset
            # times = (partial.times + preoffset)*factor+postoffset
            newpartials.append(partial.clone(times=times))
        return self.__class__(newpartials)

    def freqwarp(self, curve):
        # type: (Callable[[float], float]) -> Spectrum
        """
        curve: maps freq to freq
        """
        return Spectrum([p.freqwarp(curve) for p in self.partials])

    def freqwarp_dynamic(self, gradient):
        # type: (Callable[[float, float], float]) -> Spectrum
        """
        gradient: a callable of the form gradient(t, f) -> f
        """
        return Spectrum([p.freqwarp_dynamic(gradient) for p in self.partials])

    def shifted(self, dt=0, df=0):
        # type: (float, float) -> Spectrum
        """
        Return a new Spectrum shifted in time and/or freq. 

        dt: a constant time
        df: either a constant or dynamic (bpf) frequency
        """
        partials = [p.shifted(dt=dt, df=df) for p in self.partials]
        return Spectrum(partials)

    def __lshift__(self, secs):
        return self.shifted(-secs)

    def __rshift__(self, secs):
        return self.shifted(secs)

    def synthesize(self, samplerate=44100, outfile=""):
        # type: (int, str) -> np.ndarray
        """
        Synthesize this Spectrum

        * samplerate: the samplerate of the synthesized samples
        * outfile: if given, the samples are saved to this path. The
                   format is determined by the given extension.
                   Supported formats: wav, aif, flac.
                   wav, aif  -> flt32
                   flac      -> int24

        Returns: the samples as numpy array
        """
        matrices, labels = self.toarray()
        samples = io.synthesize(matrices, samplerate)
        if outfile:
            io.sndwrite(samples, samplerate, outfile)
        return samples

    def tosample(self, samplerate=44100):
        from emlib.snd import audiosample
        samples = self.synthesize(samplerate)
        return audiosample.Sample(samples, samplerate)

    def estimatef0(self, minfreq, maxfreq, interval=0.1):
        # type: (float, float, float) -> io.EstimateF0
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
        matrices, labels = self.toarray()
        self._f0 = io.estimatef0(matrices, minfreq, maxfreq, interval)
        return self._f0

    def normalize(self, maxpeak=1):
        # type: (float) -> Spectrum
        maxamp = max(p.amps.max() for p in self.partials)
        ratio = maxpeak / maxamp
        partials = [p.scaleamp(ratio) for p in self.partials]
        return self.__class__(partials)

    def simplify(self, pitchdelta=0.01, dbdelta=0.1, bwdelta=0.01):
        # type: (float, float, float) -> Spectrum
        """ Simplify each partial """
        newpartials = [p.simplify(pitchdelta=pitchdelta, dbdelta=dbdelta, bwdelta=bwdelta)
                       for p in self]
        return Spectrum(newpartials)


# <--------------------------------- END Spectrum


def fromarray(arrayseq, labels=None):
    # type: (List[np.ndarray], Opt[List[int]]) -> Spectrum
    """
    construct a Spectrum from array data

    arrayseq: a seq. of matrices, where each matrix is a 2D array 
              with columns -> [time, freq, amp, phase, bw]
    """
    newpartials = []
    if labels is None:
        labels = [0] * len(arrayseq)
    for matrix, label in zip(arrayseq, labels):
        times = matrix[:, 0]
        if len(times) < 1 or times[-1] - times[0] <= 0:
            logger.debug("skipping short partial")
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
    # type: (List[Spectrum], float, bool) -> Spectrum
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
    partials = []  # type: List[Partial]
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
    return Partial(p.time, p.freq, p.amp, p.phase, p.bw, label=p.label)


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
    t0 = max(f0.t0, partial.t0)
    t1 = min(f0.t1, partial.t1)
    if t1 < t0:
        raise ValueError("partial and f0 should intersect in time")
    dt = (t1 - t0) / 100
    freqs0 = f0.freq[t0:t1:dt]
    freqs1 = partial.freq[t0:t1:dt]
    prod = freqs1/freqs0
    dev = np.abs(prod - prod.round()).mean()
    return dev

