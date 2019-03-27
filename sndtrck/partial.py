from __future__ import absolute_import
from collections import namedtuple
import numpy as np
import bpf4 as bpf
from bpf4.api import BpfInterface as _Bpf
from numpyx import array_is_sorted, minmax1d, nearestitem
from numpyx import trapz as _trapz, weightedavg as _weightedavg, allequal as _allequal
from emlib.lib import snap_to_grid, snap_array
from emlib.pitchtools import amp2db, f2m, interval2ratio
from emlib.pitchtoolsnp import amp2db_np, db2amp_np, interval2ratio_np, f2m_np, m2f_np
from emlib.iterlib import pairwise
from . import typehints as t
import logging

from .const import UNSETLABEL
from .util import interpol_linear


logger = logging.getLogger("sndtrck")

OptArr = t.Opt[np.ndarray]
arr = np.ndarray

Breakpoint = namedtuple("Breakpoint", "freq amp phase bw")


_zeros = np.zeros((4000,), dtype=float)


def getzeros1(size):
    if size < _zeros.size:
        return _zeros[:size]
    return np.zeros((size,), dtype=float)


###################################################################
#
#   Partial
#
###################################################################


def _asFloatArray(a: t.U[t.Seq, np.ndarray]) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    return np.array(a, dtype=float)
    

class Partial(object):

    def __init__(self,
                 times: np.ndarray,
                 freqs: np.ndarray,
                 amps: np.ndarray,
                 phase: np.ndarray = None,
                 bw: np.ndarray = None,
                 label: int = 0,
                 meta: str = None
                 ) -> None:
        """
        label:
            a numeric label, mostly used internally to mark partials to apply transformations
        meta:
            a string to indicate any kind of metadata about this partial.
            The format used is:
            meta = "keyA=valueA;keyB=valueB;keyC=value with spaces"
        """
        self.times = T = _asFloatArray(times)   # type: np.ndarray
        L = len(T)
        if L < 2:
            if L == 0:
                raise ValueError("Creating an empty partial. Why?!")
            logger.debug("Created a Partial with only one breakpoint.\n"
                         f"times: {times}, freqs: {freqs}, amps: {amps}")
        if T[0] < 0:
            raise ValueError("A Partial can't have negative times")
        self.freqs: np.ndarray = _asFloatArray(freqs)
        self.amps: np.ndarray = _asFloatArray(amps)
        self.phases: t.Opt[np.ndarray] = None if phase is None else _asFloatArray(phase)
        self.bws: t.Opt[np.ndarray] = None if bw is None else _asFloatArray(bw)
        self.label: int = label if label is not None else UNSETLABEL
        self.t0: float = T[0]
        self.t1: float = T[-1]
        self.numbreakpoints:int = L
        self._meta: t.Opt[str] = meta
        self._freqbpf = None
        self._ampbpf = None
        self._phasebpf = None
        self._bwbpf = None
        self._array: t.Opt[np.ndarray] = None    
        self._meanfreq = -1.0
        self._meanamp = -1.0
        self._wmeanfreq = -1.0
        self._minfreq = -1.0
        self._maxfreq = -1.0
        self._hash = 0
        
    @property
    def duration(self) -> float:
        return self.t1 - self.t0
    
    @property
    def freq(self) -> float:
        if self._freqbpf is None:
            if self.numbreakpoints > 1:
                self._freqbpf = bpf.core.Linear(self.times, self.freqs)
                self.times, self.freqs = self._freqbpf.points()
            else:
                self._freqbpf = bpf.core.Const(self.freqs[0])[self.t0:self.t1]
        return self._freqbpf

    @property
    def amp(self) -> float:
        if self._ampbpf is None:
            if self.numbreakpoints > 1:
                self._ampbpf = bpf.core.Linear(self.times, self.amps)
                self.times, self.amps = self._ampbpf.points()
            else:
                self._ampbpf = bpf.core.Const(self.amps[0])[self.t0:self.t1]        
        return self._ampbpf

    @property
    def bw(self) -> float:
        if self._bwbpf is None:
            if self.numbreakpoints > 1:
                if self.bws is not None:
                    self._bwbpf = bpf.core.Linear(self.times, self.bws)
                    self.times, self.bws = self.bw.points()
                else:
                    self._bwbpf = bpf.const(0)
            else:
                bw = self.bws[0] if self.bws else 0
                self._bwbpf = bpf.core.Const(bw)
        return self._bwbpf

    def _resetcache(self):
        self.t0 = self.times[0]
        self.t1 = self.times[-1]
        self._array = None
        self._meanfreq = -1
        self._meanamp = -1
        self._wmeanfreq = -1
        self._minfreq = -1
        self._maxfreq = -1
        self._hash = 0
        self._freqbpf = None
        self._ampbpf = None
        self._bwbpf = None

    def __hash__(self):
        if self._hash > 0:
            return self._hash
        else:
            self._hash = self._calculate_hash()
            return self._hash

    def __len__(self) -> int:
        return self.numbreakpoints

    @classmethod
    def fromarray(cls, data:np.ndarray, label:int=0) -> 'Partial':
        """
        data is a 2D array with the shape (numbreakpoints, 5)
        columns: time freq amp phase bw
        """
        time, freq, amp, phase, bw = [data[:, i] for i in range(5)]
        out = cls(time, freq, amp, phase, bw, label=label)
        out._array = data
        return out

    def toarray(self) -> np.ndarray:
        """
        Convert this Partial to a 2D array
        """
        array = self._array
        if array is not None:
            return array
        t = self.times
        f = self.freqs
        a = self.amps
        numbp = t.shape[0]
        if self.phases is not None:
            ph = self.phases
        else:
            self.phases = ph = getzeros1(numbp)
        if self.bws is not None:
            bw = self.bws
        else:
            self.bws = bw = getzeros1(numbp)
        self._array = array = np.column_stack((t, f, a, ph, bw))
        return array

    def __getitem__(self, i:int) -> np.ndarray:
        a = self._array
        return a[i] if a is not None else self.toarray()[i]
        
    def __iter__(self):
        return iter(self.toarray())   

    def __repr__(self):
        return "Partial %d [%.4f:%.4f] %.1fHz" % \
            (self.label, self.t0, self.t1, self.meanfreq)
        
    def dump(self, showdb=True) -> str:
        lines = [f"Partial {self.label} [{self.t0:.4f}:{self.t1:.4f}]"]
        if self.getmeta():
            lines.append(f"   metadata: {self._meta}")
        a = self.toarray()
        if showdb:
            a = a.copy()
            a[:,2] = np.round(amp2db_np(a[:,2]), 1)
        arraystr = np.array2string(a, precision=3, suppress_small=True, separator=" ")
        lines.append(arraystr)
        return "\n".join(lines)

    def setmeta(self, key:str, value:str) -> None:
        d = self.getmeta()
        d[key] = str(value)
        items = ("=".join((key, value)) for key, value in d.items())
        metastr = ";".join(items)
        self._meta = metastr

    def getmeta(self) -> t.Dict[str, str]:
        """
        Returns a dictionary representing the metadata of this partial
        * this dictionary is read only, any modifications to it will NOT
          be reflected back in the partial
        * in order to modify the metadata, use .setmeta(key, value)
        * all values are converted to strings
        """
        if not self._meta:
            return {}
        pairs = [pair.split("=") for pair in self._meta.split(";")]
        return {pair[0]:pair[1] for pair in pairs}

    def __eq__(self, other:'Partial') -> bool:
        if self is other:
            return True
        if self.numbreakpoints != other.numbreakpoints:
            return False
        if not _allequal(self.times, other.times):
            return False
        if not _allequal(self.freqs, other.freqs):
            return False
        if not _allequal(self.amps, other.amps):
            return False
        return True

    def _calculate_hash(self) -> int:
        data = self.times * self.amps
        return hash(data.tostring())

    def __ne__(self, other) -> bool:
        return not(self == other)

    def at(self, t:float) -> 'Breakpoint':
        """
        returns a Breakpoint interpolated at time `t`
        """
        return Breakpoint(self.freq(t), self.amp(t), self.phase(t), self.bw(t))

    def _insert_many(self, times, freqs, amps, bws=None):
        assert len(times) == len(freqs) == len(amps)
        idx = np.searchsorted(self.times, times)
        times2 = np.insert(self.times, idx, times)
        freqs2 = np.insert(self.freqs, idx, freqs)
        amps2 = np.insert(self.amps, idx, amps)
        if self.bws is None and bws is None:
            bws2 = None
        else:
            bws0 = self.bws if self.bws is not None else getzeros1(len(self.times))
            bws = bws if bws is not None else getzeros1(len(times))
            bws2 = np.insert(bws0, idx, bws)
        return Partial(times2, freqs2, amps2, bw=bws2, label=self.label)

    def insert(self,
               time: t.U[float, t.Seq[float]],
               freq: t.U[float, t.Seq[float]],
               amp: t.U[float, t.Seq[float]],
               bw=0) -> 'Partial':
        """
        Insert a breakpoint at given time. Returns a **new** Partial

        time: a time or a seq. of times
        freq: a freq or a seq. of freqs
        amp: an amp or a seq. of amps

        NB: If you plan to insert many points at once, it is more efficient to build 
            first a seq. of breakpoints and  then call partial.insert(times, freqs, amps)
        """
        if isinstance(time, float):
            time = [time]
            freq = [freq]
            amp = [amp]
            bw = [bw]
        return self._insert_many(time, freq, amp, bw)

    def append(self, time, freq, amp, bw=0):
        """
        Append a Breakpoint to a COPY of this Partial (Partials are immutable)

        This is NOT an efficient opperation since all data has
        to be copyied. If you want to build a Partial from a seq.
        of breakpoints, considere first building an array
        of times, freqs, amps, etc. and then building a Partial
        from that

        time: a time or a seq. of times (in this case, the other params
              should also be a seq. of the same size)
        freq: a freq or a seq. of freqs
        amp: an amp or a seq of amps
        bw: a bw or a seq. of bws
        """
        if isinstance(time, float):
            time = [time]
        assert time[0] > self.t1    
        freqs = np.append(self.freqs, freq)
        amps = np.append(self.amps, amp)
        times = np.append(self.times, time)
        bws = np.append(self.bws, bw) if self.bws else None
        return Partial(times, freqs, amps, phase=None, bws=bws, label=self.label)
   
    def crop(self, t0, t1):
        # type: (float, float) -> Partial
        """
        Returns a new Partial cropped to t0-t1

        See also: slice
        """
        # TODO: make this faster
        if t0 < self.t0 and self.t1 < t1:
            return self
        t0 = max(t0, self.t0)
        t1 = min(t1, self.t1)
        assert t1 > t0
        times, freqs = bpf.util.arrayslice(t0, t1, self.times, self.freqs)
        amps = self.amp.map(times)
        phases = self.phase.map(times) if self.phases is not None else None
        bws = self.bw.map(times) if self.bws is not None else None
        return self.__class__(times, freqs, amps, phases, bws, label=self.label)

    def slice(self, t0:float, t1:float, include=True) -> 'Partial':
        """
        Slice this partial without adding new breakpoints.
        In comparison to crop, slice returns a "view" to this 
        partial between the given times. 

        If include is true, the view extends over the range t0-t1,
        slicing at the first breakpoint after the given time

        If include is false, the slice is produced at the last
        breakpoint within the given range
        """
        if t0 > self.t1 or t1 < self.t0:
            raise ValueError("Partial not defined within the slice range")
        if t0 < self.t0 and self.t1 < t1:
            return self
        times = self.times
        idx0 = times.searchsorted(t0)
        idx1 = times.searchsorted(t1)
        if include:
            idx0 = max(0, idx0 - 1)
            idx1 += 1
        T = times[idx0:idx1]
        F = self.freqs[idx0:idx1]
        A = self.amps[idx0:idx1]
        B = self.bws[idx0:idx1]
        P = self.phases[idx0:idx1]
        return Partial(T, F, A, P, B)

    def copy(self) -> 'Partial':
        out = Partial(self.times, self.freqs, self.amps, self.phases, self.bws,
                      label=self.label, meta=self._meta)
        out._array = self._array
        out._meanamp = self._meanamp
        out._meanfreq = self._meanfreq
        out._wmeanfreq = self._wmeanfreq
        return out
    
    def clone(self,
              times: np.ndarray = None,
              freqs: np.ndarray = None,
              amps: np.ndarray = None,
              phases: np.ndarray = None,
              bws: np.ndarray = None,
              label: int = None,
              meta: str = None) -> 'Partial':
        """
        Clone this Partial, optionally overriding some of its data

        >>> B = A.clone(amps=A.amps*2)
        """
        T = times if times is not None else self.times
        F = freqs if freqs is not None else self.freqs
        A = amps if amps is not None else self.amps
        P = phases if phases is not None else self.phases
        B = bws if bws is not None else self.bws
        label = label if label is not None else self.label
        meta = meta if meta is not None else self._meta
        out = self.__class__(T, F, A, P, B, label=label, meta=meta)
        return out

    @property
    def meanfreq(self):
        # type: () -> float
        if self._meanfreq >= 0:
            return self._meanfreq
        if self.numbreakpoints < 2:
            return self.freqs[0] if self.numbreakpoints else 0
        # the next line is 10x faster than accel.trapz! why?
        self._meanfreq = a = self.freq.integrate() / (self.t1 - self.t0)  
        # self._meanfreq = a = _trapz(self.freqs, self.times) / (self.t1 - self.t0)
        return a

    @property
    def meanfreq_weighted(self) -> float:
        """
        weighted mean frequency
        """
        if self._wmeanfreq > 0:
            return self._wmeanfreq
        if self.numbreakpoints < 2:
            return self.freqs[0] if self.numbreakpoints else 0
        self._wmeanfreq = a = _weightedavg(self.freqs, self.times, self.amps)
        return a
    
    @property
    def meanamp(self) -> float:
        if self._meanamp >= 0:
            return self._meanamp
        # self._meanamp = a = _trapz(self.amps, self.times) / (self.t1 - self.t0)
        if self.numbreakpoints < 2:
            return self.freqs[0] if self.numbreakpoints else 0
        self._meanamp = a = self.amp.integrate() / (self.t1 - self.t0)
        return a

    def energy(self) -> float:
        """
        Returns the energy provided by this partial

        energy is the integration of amplitude over time
        """
        return self.amp.integrate()

    @property
    def meanbw(self):
        if self.numbreakpoints < 2:
            return self.bws[0] if self.numbreakpoints else 0
        return _trapz(self.bws, self.times) / (self.t1 - self.t0)
    
    @property
    def minfreq(self) -> float:
        if self._minfreq >= 0:
            return self._minfreq
        f0, f1 = minmax1d(self.freqs)
        self._minfreq, self._maxfreq = minmax1d(self.freqs)
        return self._minfreq
        
    @property
    def maxfreq(self):
        # type: () -> float
        if self._maxfreq >= 0:
            return self._maxfreq
        f0, f1 = minmax1d(self.freqs)
        self._minfreq, self._maxfreq = minmax1d(self.freqs)
        return self._maxfreq

    def _scalevaramp(self, gainfunc: t.U[_Bpf, t.Fun]) -> 'Partial':
        times = self.times
        if isinstance(gainfunc, _Bpf):
            gains = gainfunc.map(times)
        else:
            gains = list(map(gainfunc, times))
            gains = np.array(gains, dtype=float)
        amps = self.amps * gains
        return self.clone(amps=amps)
        
    def gain(self, gain):
        # type: (U[Fun, _Bpf, float]) -> Partial
        """
        Return a new Partial with scaled amplitude
        
        gain: a scalar or a **callable**. If it is a callable, it will be sampled
              at the X points on this partial. 
        """
        if callable(gain):
            return self._scalevaramp(gain) 
        return self.clone(amps=self.amps*gain)

    def oversampled(self, n: int) -> 'Partial':
        dt = self.duration / (self.numbreakpoints*n)
        return self.resampled(dt)

    def fadecurve(self, time, kind='fadein', shape='linear', oversample=4):
        # type: (float, str, str, int) -> Partial
        """
        Fade the partial with the given curve

        time: fade time
        kind: one of 'fadein', 'fadeout', 'both'
        shape: the fade shape as string definition ('linear', 'expon(2)', etc)
        """
        if time < self.duration:
            raise ValueError(f"fadetime ({time}) can't be > dur. ({self.duration}) of the partial")
        
        if kind == 'fadein':
            faded = self.crop(self.t0, self.t0+time)
            curve = bpf.util.makebpf(shape, [faded.t0, faded.t1], [0, 1])
            if shape != 'linear':
                faded = faded.oversampled(oversample)
            faded = faded.scaleamp(curve.map)
            rest = self.crop(self.t0+time+0.001, self.t1)
            return concat([faded, rest])
        elif kind == 'fadeout':
            faded = self.crop(self.t1-time, self.t1)
            curve = bpf.util.makebpf(shape, [faded.t0, faded.t1], [1, 0])
            if shape != 'linear':
                faded = faded.oversampled(oversample)
            faded = faded.scaleamp(curve.map)
            rest = self.crop(self.t0, self.t1-time-0.001)
            return concat([rest, faded])
        elif kind == 'both':
            return self.fade(time, "fadein", shape).fade(time, "fadeout", shape)
        else:
            raise ValueError("kind should be 'fadein', 'fadeout' or 'both'")

    def estimate_breakpoint_gap(self, percentile=50) -> float:
        T = self.times
        diffs = T[1:] - T[:-1]
        gap = np.percentile(diffs, percentile)
        return gap
        
    def fade(self, fadetime):
        """
        Apply a fadein-out to this partial of the given fadetime.
        A fade always extends the partial

        NB: for the case where a partial begins at a time < fadetime, 
            the fadetime is shortened accordingly. If a partial begins
            at t0==0, as an exception, the fade is applied "within"
            the partial
        """
        def extendarray(a, v0, v1):
            assert v0 < a[0] and a[-1] < v1
            newa = np.empty((len(a)+2,), dtype=float)
            newa[0] = v0
            newa[-1] = v1
            newa[1:-1] = a
            return newa

        fade0 = fadetime if self.t0 > fadetime else self.t0
        if self.t0 == 0:
            npconcat = np.concatenate
            bp1 = self.times[1]
            t0 = min(fadetime, bp1*0.5)
            times2 = npconcat(([0, t0], self.times[1:], [self.t1 + fadetime]))
            assert array_is_sorted(times2)
            a0 = self.amp(t0)
            amps2 = npconcat(([0, a0], self.amps[1:], [0]))
            freqs2 = npconcat(([self.freq(0), self.freq(t0)], self.freqs[1:], [self.freqs[-1]]))
            bws2 = npconcat(([self.bws[0]], self.bws, [self.bws[-1]]))
            phases2 = None    # TODO: fix phases
        else:
            amps2 = extendarray(self.amps, 0, 0)
            times = self.times
            times2 = extendarray(times, times[0]-fade0, times[-1]+fadetime)
            freqs = self.freqs 
            freqs2 = extendarray(freqs, freqs[0], freqs[-1])
            phases2 = None    # TODO: fix phases
            bws = self.bws
            bws2 = extendarray(bws, bws[0], bws[-1])
        return Partial(times2, freqs2, amps2, phases2, bws2)
        
    def timewarp(self, curve):
        # type: (_Bpf) -> Partial
        """
        Warp the time of this Partial according to the curve.
        
        curve: bpf mapping the time of this partial to new times,
               possibly in a non-linear way
        
        Returns a new Partial

        NB: the resulting Partial will have the same amount of breakpoints
            as the original. To allow for a more detailed mapping, resampling
            might be needed
        """
        if self.numbreakpoints <= 2:
            times1 = [curve(t) for t in self.times]
        else:
            times0 = self.times
            times1 = curve.map(times0)
        return self.clone(times=times1)
                
    def freqwarp(self, curve:_Bpf) -> 'Parial':
        """
        Example: snap this spectrum to the chromatic scale

        def snap(freq):
            return m2f(round(f2m(freq)))

        curve = bpf.asbpf(snap)
        snapped_partial = partial.freqwarp(curve)

        :param curve: Bpf. maps freq -> freq
        :return:
        """
        freqs = self.freqs
        freqs = curve.map(freqs)
        return self.clone(freqs=freqs)

    def freqwarp_dynamic(self, gradient:t.Fun[[float, float], float]) -> 'Partial':
        """
        Similar to freqwarp, but the warping curve can change over time,
        so it takes a gradient (a function of time and freq)

        gradient: a Gradient mapping (time, freq) -> freq
        """
        freqs2 = np.fromiter((gradient(t, f) for t, f in zip(self.times, self.freqs)), 
                             dtype=float)
        return self.clone(freqs=freqs2)

    def equalize(self, curve:_Bpf) -> 'Partial':
        """
        Apply a frequency dependent gain to each breakpoint

        curve: a bpf mapping freq to gain
        
        Returns a new Partial with modified amplitudes
        """
        amps = self.amps * curve.map(self.freqs)
        return self.clone(amps=amps)

    def normalized(self, maxpeak=1.0) -> 'Partial':
        """
        Return a normalized version of this Partial

        Example:

        normalized_partial = partial.normalized(db2amp(-3))
        """
        X, Y = self.amp.points()
        gain = maxpeak / np.absolute(Y).max()
        return self.scaleamp(gain)
        
    def resampled(self, dt:float) -> 'Partial':
        """
        Return a new partial resampled at the given time interval

        dt: the sampling interval
        
        Returns the resampled Partial
        """
        # N = (self.t1 - self.t0) / dt
        if len(self.times) == 1:
            logger.debug("resampling a partial with only one breakpoint")
            times = np.array([snap_to_grid(self.times[0], dt)])
            return self.clone(times=times)
        
        t0 = snap_to_grid(self.t0, dt)
        t1 = max(snap_to_grid(self.t1, dt), t0+dt)
        times = np.arange(t0, t1+dt/2, dt)
        if len(times) > 2:
            assert (len(times)-1)/(t1-t0) <= (1/dt)+1, f"t0:{t0} t1:{t1} dt:{dt} times:{times}"
        freqs = self.freq.map(times)
        amps = self.amp.map(times)
        phases = None if self.phases is None else self.phase.map(times) 
        bws = None if self.bws is None else self.bw.map(times)
        return Partial(times, freqs, amps, phases, bws, label=self.label)

    def transpose(self, interval: t.U[float, t.Fun1]) -> 'Spectrum':
        if isinstance(interval, (int, float)):
            ratio = interval2ratio(interval)
            return self.clone(freqs=self.freqs*ratio)
        elif callable(interval):
            intervals = bpf.asbpf(interval).map(self.times)
            ratios = interval2ratio_np(intervals)
            return self.clone(freqs=self.freqs*ratios)
        else:
            raise TypeError("interval should be a number or a function(time) -> interval")
        
    def shifted(self, dt=0.0, df:t.U[float, t.Fun1]=0.0) -> 'Partial':
        """
        Shift this partial on either the time or freq coord.
        To shift frequency by a given interval, use transpose

        dt: delta time
        df: delta freq, can be a function (time) -> df
        
        Example: shift a Partial 0.5 seconds later, and 100 Hz higher

        >> partial.shifted(0.5, 100)
        """
        if self.t0 + dt < 0:
            raise ValueError("a Partial can't have a negative time")
        freqs = self.freqs
        times = self.times
        if df != 0:
            if isinstance(df, (int, float)):
                freqs = freqs + df  # type: np.ndarray
                if (freqs < 0).any():
                    raise ValueError("a Partial can't have negative frequencies")
            elif callable(df):
                freqs = freqs + bpf.asbpf(df).map(times)
                if (freqs < 0).any():
                    raise ValueError("a Partial can't have negative frequencies")
            else:
                raise TypeError("df should be a number or a function(time) -> df")
        if dt != 0:
            times = times + dt
        return Partial(times, freqs, self.amps, self.phases, self.bws, label=self.label)
        
    def quantized(self,
                  pitchgrid: t.U[float, np.ndarray]=None,
                  dbgrid:t.U[float, np.ndarray]=None) -> 'Partial':
        """
        Returns a new partial quantized to the given grids

        pitchgrid: the resolution of the pitch grid, or a seq. of possible pitches
                    (fractional midinotes). Use None to skip freq. quantization
        dbgrid: the resolution of the amp grid, in dB, or a list of possible amps,
                 in dB. Use None to not quantize amps
        
        * NB1: call .simplified to remove redundant breakpoints after quantization
        * NB2: to generate a list of dB as expected in dbgrid from a dynamic curve 
               call dyncurve.todbs()
        """
        assert pitchgrid is not None or dbgrid is not None
        if pitchgrid is not None:
            midis = f2m_np(self.freqs)  
            if isinstance(pitchgrid, (int, float)):
                dp = pitchgrid
                midis = snap_array(midis, dp, out=midis)
            elif isinstance(pitchgrid, (np.ndarray, list, tuple)):
                midis = nearestitem(np.asarray(pitchgrid), midis)    
            else:
                raise TypeError(f"pitchgrid has an unknown type: {type(pitchgrid)}")
            freqs = m2f_np(midis, out=midis)
        else:
            freqs = self.freqs
        if dbgrid is not None:
            dbs = amp2db_np(self.amps)
            if isinstance(dbgrid, (int, float)):
                dbs = snap_array(dbs, dbgrid, out=dbs)
            elif isinstance(dbgrid, (np.ndarray, list, tuple)):
                dbs = nearestitem(np.asarray(dbgrid), dbs)
            else:
                raise TypeError(f"dbgrid expects number or seq. of numbers, got: {type(dbgrid)}")
            amps = db2amp_np(dbs, out=dbs)
        else:
            amps = self.amps
        return self.clone(freqs=freqs, amps=amps)

    def simplify(self, pitchdelta=0.1, dbdelta=1.0, bwdelta=0.1) -> 'Partial':
        """
        Remove breakpoints where the variation is under the given deltas. 
        A breakpoint is removed if the variation it imposes is below any
        of the given deltas. 

        NB: In order to remove breakpoints based on pitch
        alone, dbdelta and bwdelta should be set very high

        Returns a new Partial
        """
        T, F, A, B = self.times, self.freqs, self.amps, self.bws
        if B is None:
            B = getzeros1(T.size)
        breakpoints = list(zip(T, F, A, B))
        newbreakpoints = [breakpoints[0]]
        i = 0
        lastindex = len(breakpoints) - 1
        # i, para cada breakpoint que sigue,
        while i <= lastindex - 2:
            j = i + 1
            foundvalid = False
            while j <= lastindex - 1:
                b0 = breakpoints[i]
                b1 = breakpoints[j]
                b2 = breakpoints[j+1]
                if _breakpoint_deviates(b0, b1, b2, dbdelta=dbdelta, 
                                        pitchdelta=pitchdelta, bwdelta=bwdelta):
                    newbreakpoints.append(b1)
                    i = j
                    foundvalid = True
                    break
                else:
                    j += 1
            if not foundvalid:
                break
        newbreakpoints.append(breakpoints[-1])
        T, F, A, B = zip(*newbreakpoints)
        return Partial(T, F, A, bw=B, label=self.label)

    def harmonicdev(self, f0:'Partial') -> float:
        """
        How much does partial deviate from being a harmonic of f0?

        Returns: a value between 0 and 0.5

        TODO: take correlation into account
        """
        t0 = max(f0.t0, self.t0)
        t1 = min(f0.t1, self.t1)
        if t1 < t0:
            raise ValueError("partial and f0 should intersect in time")
        dt = (t1 - t0) / 100
        freqs0 = f0.freq[t0:t1:dt]
        freqs1 = self.freq[t0:t1:dt]
        prod = freqs1/freqs0
        dev = np.abs(prod - prod.round()).mean()
        return dev


def _partials_overlap(partials):
    for p0, p1 in pairwise(partials):
        if p0.t1 >= p1.t0:
            return True
    return False


def concat2(partials, fade=0.005):
    # type: (Seq[Partial], float) -> Partial
    """
    Concatenate multiple Partials to produce a new one.
    Assumes that the partials are non-overlapping and sorted

    partials: a seq. of Partials
    fade: fade time (both fade-in and fade-out) in the case that
          the partials don't begin or end with a 0 amp. Fade time always
          extends the partial. The partials must have a gap between
          them gap > 2*fade. If the gap is less than that, the
          second partial will be cropped.
    """
    # fade = max(fade, 128/48000.)
    if _partials_overlap(partials):
        for p in partials:
            print(p)
        raise ValueError("partials overlap, can't be concatenated")
    numpartials = len(partials)
    if numpartials == 0:
        raise ValueError("No partials to concatenate")
    T, F, A, B = [], [], [], []
    minfade = 0.001    # min. fade
    fade0 = fade
    fade = max(minfade, fade - 2*minfade)
    zeros = np.zeros((4,), dtype=float)
    zero = np.zeros((1,), dtype=float)

    assert fade > 0

    p = partials[0]
    t0 = max(0, p.times[0] - fade)
    if p.times[0] > 0:
        T.append([t0])
        F.append([p.freqs[0]])
        A.append(zero)
        B.append(zero)
    else:
        times = p.times
        bp1_t = min(t for t in [fade, (times[1]-times[0])*0.5] if t > 0)
        T.append([0, bp1_t])
        F.append([p.freqs[0], p.freq(bp1_t)])
        A.append([0, p.amp(bp1_t)])
        B.append([p.bws[0], p.bw(bp1_t)])

    if numpartials == 1:
        p = partials[0]
        T.append(p.times)
        F.append(p.freqs)
        A.append(p.amps)
        B.append(p.bws)
    else:
        for p0, p1 in pairwise(partials):
            if p0.t1 > p1.t0:
                raise ValueError("Partials are overlapping, cannot concatenate")
            t0 = p0.times[-1]
            t1 = p1.times[0]
            assert p1.times[0] - p0.times[-1] > fade0*2
            T.append(p0.times)
            F.append(p0.freqs)
            A.append(p0.amps)
            bws = p0.bws
            bws = bws if bws is not None else np.zeros((len(p0.amps),), dtype=float) 
            B.append(bws)
            f0 = p0.freqs[-1]
            f1 = p1.freqs[0]
            assert t0+fade < t1-fade-minfade*2
            middlet = (t0+t1)*0.5
            T.append([t0+fade, middlet - minfade, middlet+minfade, t1-fade])
            F.append([f0,      f0,            f1,          f1])
            A.append(zeros)
            B.append(zeros)
        T.append(p1.times)
        F.append(p1.freqs)
        A.append(p1.amps)
        bws = p1.bws
        bws = bws if bws is not None else np.zeros((len(p0.amps),), dtype=float) 
        B.append(bws) 
    
    p = partials[-1]
    if p.amps[-1] > 0:
        t1 = p.times[-1]
        f1 = p.freqs[-1]
        T.append([t1 + fade])
        F.append([f1])
        A.append(zero)
        B.append(zero)

    times = np.concatenate(T)
    freqs = np.concatenate(F)
    amps = np.concatenate(A)
    bws = np.concatenate(B)
    assert array_is_sorted(times), times
    return Partial(times, freqs, amps, bw=bws)

def concat(partials, fade=0.005):
    # type: (Seq[Partial], float) -> Partial
    """
    Concatenate multiple Partials to produce a new one.
    Assumes that the partials are non-overlapping and sorted

    partials: a seq. of Partials
    fade: fade time (both fade-in and fade-out) in the case that
          the partials don't begin or end with a 0 amp. Fade time always
          extends the partial. The partials must have a gap between
          them gap > 2*fade. If the gap is less than that, the
          second partial will be cropped.
    """
    minfade = 0.001
    fade = max(minfade, fade)
    T, F, A, B = [], [], [], []
    now = -fade

    def newbp(t, f, a, bw):
        nonlocal now
        assert t > now
        t = max(t, 0)
        T.append([t])
        F.append([f])
        A.append([a])
        B.append([bw])
        now = t

    def append_data(ts, fs, amps, bws):
        nonlocal now
        assert ts[0] > now, f"now: {now}, times: {ts}"
        T.append(ts)
        F.append(fs)
        A.append(amps)
        B.append(bws)
        now = ts[-1]

    def append_partial(p):
        append_data(p.times, p.freqs, p.amps, p.bws)

    def concat_partial(p):
        if p.amps[0] > 0 and p.times[0] > 0:
            t0 = p.times[0] - fade 
            newbp(t0, p.freqs[0], 0, 0)

        append_partial(p)
        if p.amps[-1] > 0:
            t1 = p.times[-1] + fade
            newbp(t1, p.freqs[-1], 0, 0)
    
    for p in partials:
        concat_partial(p)
    
    times = np.concatenate(T)
    freqs = np.concatenate(F)
    amps = np.concatenate(A)
    bws = np.concatenate(B)
    assert array_is_sorted(times), times
    return Partial(times, freqs, amps, bw=bws)


# -------------------------- utilities ----------------------------

def _breakpoint_deviates(b0, b1, b2, dbdelta, pitchdelta, bwdelta):
    """
    True if b1 deviates from the interpolation of b0 and b2
    
    bx: a tuplet of (time, freq, amp, bw)
    """
    t0, f0, a0, bw0 = b0
    t, f, a, bw = b1
    t1, f1, a1, bw1 = b2
    a_t = interpol_linear(t, t0, a0, t1, a1)
    varamp = abs(amp2db(a) - amp2db(a_t))
    if varamp >= dbdelta:
        return True
    f_t = interpol_linear(t, t0, f0, t1, f1)
    varpitch = abs(f2m(f_t) - f2m(f))
    if varpitch >= pitchdelta:
        return True
    bw_t = interpol_linear(t, t0, bw0, t1, bw1)
    if abs(bw_t - bw) >= bwdelta:
        return True
    return False
