from __future__ import absolute_import
from collections import namedtuple
from numbers import Number as _Number
import numpy as np
import bpf4 as bpf
from .const import UNSETLABEL
from emlib.lib import isiterable, nearest_element
from .util import interpol_linear
from emlib.pitch import amp2db, db2amp, f2m, m2f
from emlib.iters import pairwise
from typing import Optional as Opt, Callable, Union, Tuple, Sequence as Seq

OptArr = Opt[np.ndarray]
arr = np.ndarray

Breakpoint = namedtuple("Breakpoint", "freq amp phase bw")

PartialData = namedtuple("PartialData", ["times", "freqs", "amps", "phases", "bws"])


Bpf = bpf.BpfInterface

###################################################################
#
#   Partial
#
###################################################################


class Partial(object):

    def __init__(self, times, freqs, amps, phase=None, bw=None, label=0):
        # type: (np.ndarray, np.ndarray, np.ndarray, Opt[np.ndarray], Opt[np.ndarray], int) -> None
        assert isiterable(times)
        if times[0] < 0:
            raise ValueError("A Partial can't have negative times")
        self.times = times   # type: np.ndarray
        self.freqs = freqs   # type: np.ndarray
        self.amps = amps     # type: np.ndarray
        self.phases = phase  # type: Opt[np.ndarray]
        self.bws = bw        # type: Opt[np.ndarray]
        self._freqbpf = None
        self._ampbpf = None
        self._phasebpf = None
        self._bwbpf = None
        self.label = label if label is not None else UNSETLABEL  # str
        self._array = None         # type: Opt[np.ndarray]
        self._meanfreq = -1        # type: float
        self._meanamp = -1         # type: float
        self._wmeanfreq = -1       # type: float
        self._minfreq = -1         # type: float
        self._maxfreq = -1         # type: float
        self.t0 = self.times[0]    # type: float
        self.t1 = self.times[-1]   # type: float
        self._hash = 0             # type: int
        self.numbreakpoints = len(times)   # type: int
    
    @property
    def duration(self):
        # type: () -> float
        return self.t1 - self.t0
    
    @property
    def freq(self):
        f = self._freqbpf 
        if f is None:
            self._freqbpf = f = bpf.core.Linear(self.times, self.freqs)
        return f

    @property
    def amp(self):
        a = self._ampbpf
        if a is None:
            self._ampbpf = a = bpf.core.Linear(self.times, self.amps)
        return a

    @property
    def phase(self):
        p = self._phasebpf
        if p is None:
            if self.phases is not None:
                self._phasebpf = p = bpf.core.Linear(self.times, self.phases)
            else:
                self._ohasebpf = p = bpf.const(0)
        return p

    @property
    def bw(self):
        bw = self._bwbpf
        if bw is None:
            if self.bws is not None:
                self._bwbpf = bw = bpf.core.Linear(self.times, self.bws)
            else:
                self._bwbpf = bw = bpf.const(0)
        return bw

    def _resetcache(self):
        self._array = None
        self._meanfreq = -1
        self._meanamp = -1
        self._wmeanfreq = -1
        self._minfreq = -1
        self._maxfreq = -1
        self.t0 = self.times[0]
        self.t1 = self.times[-1]
        self._hash = 0

    def __hash__(self):
        if self._hash > 0:
            return self._hash
        else:
            self._hash = self._calculate_hash()
            return self._hash

    def __len__(self):
        # type: () -> int
        return self.numbreakpoints

    def freqrange(self):
        # type: () -> Tuple[float, float]
        freqs = self.getdata().freqs
        return freqs.min(), freqs.max()

    @classmethod
    def fromarray(cls, data, label=0):
        # type: (np.ndarray, int) -> Partial
        """
        data is a 2D array with the shape (numbreakpoints, 5)
        columns: time freq amp phase bw
        """
        time, freq, amp, phase, bw = [np.ascontiguousarray(data[:, i]) for i in range(5)]
        out = cls(time, freq, amp, phase, bw, label=label)
        out._array = data
        return out

    def toarray(self):
        # type: () -> np.ndarray
        if self._array is not None:
            return self._array
        t = self.times
        f = self.freqs
        a = self.amps
        ph = self.phases if self.hasphase() else np.zeros_like(t)
        bw = self.bws if self.hasbandwidth() else np.zeros_like(t)
        self._array = a = np.column_stack((t, f, a, ph, bw))
        return a

    def __getitem__(self, i):
        # type: (int) -> np.ndarray
        a = self._array
        return a[i] if a is not None else self.toarray()[i]
        
    def hasbandwidth(self):
        # type: () -> bool
        return self.bws is not None
    
    def hasphase(self):
        # type: () -> bool
        return self.phases is not None

    def __iter__(self):
        return self.toarray()    

    def __repr__(self):
        return "Partial %d [%.4f:%.4f] %.1fHz" % (self.label, self.t0, self.t1, self.meanfreq_weighted)
        
    def dump(self):
        lines = ["Partial %d [%.4f:%.4f]" % (self.label, self.t0, self.t1)]
        a = self.toarray().copy()
        a[:,2] = np.round(amp2db(a[:,2]), 1)
        arraystr = np.array2string(a, precision=3, suppress_small=True, separator=" ")
        lines.append(arraystr)
        return "\n".join(lines)

    def __eq__(self, other):
        # type: (Partial) -> bool
        if self is other:
            return True
        if self.numbreakpoints != other.numbreakpoints:
            return False
        if self.times is not other.times and (self.times != other.times).any():
            return False
        if (self.freqs != other.freqs).any():
            return False
        if (self.amps != other.amps).any():
            return False
        return True

    def _calculate_hash(self):
        data = self.times * self.amps
        return hash(data.tostring())

    def __ne__(self, other):
        return not(self == other)

    def data_at(self, t):
        # type: (float) -> Breakpoint
        return Breakpoint(self.freq(t), self.amp(t), self.phase(t), self.bw(t))

    def getdata(self):
        # type: () -> PartialData
        """
        Returns (times, freq, amps, phases, bws)

        * times, freqs and amps are mandatory and will be numpy arrays
        * phases and bws can be either a numpy array or None
        """
        return PartialData(self.times, self.freqs, self.amps, self.phases, self.bws)

    def insert(self, time, freq, amp):
        # type: (float, float, float) -> Partial
        """
        Insert a breakpoint at given time. Returns a **new** Partial

        time: a time or a seq. of times
        freq: a freq or a seq. of freqs
        amp: an amp or a seq. of amps

        NB: If you plan to insert many points at once, it is better
            to call insert with a list of breakpoints
            partial.insert(times, freqs, amps)
        """
        return self.insertmany([time], [freq], [amp])

    def insertmany(self, times, freqs, amps):
        # type: (Seq[float], Seq[float], Seq[float]) -> Partial
        """
        Insert many breakpoints at once. Returns a NEW Partial
        :param times: a seq. of times
        :param freqs: a seq. of freqs
        :param amps: a seq. of amps
        :return: a new Partial
        """
        assert len(times) == len(freqs) == len(amps)
        data = self.toarray()
        for t, f, a in zip(times, freqs, amps):
            idx = np.searchsorted(data[:,0], t)
            data = np.insert(data, idx, [t, f, a, 0, 0], axis=0)
        return Partial.fromarray(data, label=self.label)

    def crop(self, t0, t1):
        # type: (float, float) -> Partial
        """Returns a new Partial cropped to t0-t1"""
        # TODO: make this faster
        t0 = max(t0, self.t0)
        t1 = min(t1, self.t1)
        assert t1 > t0
        times, freqs = self.freq.sliced(t0, t1).points()
        amps = self.amp.map(times)
        phases = self.phase.map(times) if self.phases is not None else None
        bws = self.bw.map(times) if self.bws is not None else None
        return self.__class__(times, freqs, amps, phases, bws, label=self.label)

    def copy(self):
        # type: () -> Partial
        out = self.__class__(self.times, self.freqs, self.amps, self.phases, self.bws, 
                             label=self.label)
        out._array = self._array
    
    def clone(self, times=None, freqs=None, amps=None, phases=None, bws=None, copyold=False):
        # type: (Opt[np.ndarray], Opt[np.ndarray], Opt[np.ndarray], Opt[np.ndarray], Opt[np.ndarray], bool) -> Partial
        """
        Clone this Partial, optionally overriding some of its data

        >>> B = A.clone(amps=A.amps*2)
        """
        T, F, A, P, B = self.getdata()
        _ = lambda a:a.copy() if a is not None and copyold else a
        T = times or _(T)
        F = freqs or _(F)
        A = amps or _(A)
        P = phases or _(P)
        B = bws or _(B)
        return self.__class__(T, F, A, P, B)

    @property
    def meanfreq(self):
        # type: () -> float
        if self._meanfreq >= 0:
            return self._meanfreq
        self._meanfreq = mean = self.freq.integrate() / (self.t1 - self.t0)
        return mean

    @property
    def meanfreq_weighted(self):
        # type: () -> float
        """
        weighted mean frequency
        """
        if self._wmeanfreq > 0:
            return self._wmeanfreq
        sumweight = self.amp.integrate()
        if sumweight > 0:
            out = (self.freq * self.amp).integrate() / sumweight
        else:
            out = 0
        self._wmeanfreq = out
        return out

    @property
    def meanamp(self):
        # type: () -> float
        if self._meanamp >= 0:
            return self._meanamp
        self._meanamp = self.amp.integrate() / (self.t1 - self.t0)
        return self._meanamp

    @property
    def minfreq(self):
        # type: () -> float
        if self._minfreq >= 0:
            return self._minfreq
        self._minfreq = f = self.freqs.min()
        return f

    @property
    def maxfreq(self):
        # type: () -> float
        if self._maxfreq >= 0:
            return self._maxfreq
        self._maxfreq = f = self.freqs.max()
        return f

    def _scalevaramp(self, gainfunc):
        # type: (Union[bpf.BpfInterface, Callable]) -> Partial
        times = self.times
        if isinstance(gainfunc, Bpf):
            gains = gainfunc.map(times)
        else:
            gains = list(map(gainfunc, times))
            gains = np.array(gains, dtype=float)
        amps = self.amps * gains
        return self.clone(amps=amps)
        
    def scaleamp(self, gain):
        # type: (Union[Callable, bpf.BpfInterface, float]) -> Partial
        """
        Return a new Partial with scaled amplitude
        
        gain: a scalar or a **callable**. If it is a callable, it will be sampled
              at the X points on this partial. 
        """
        if callable(gain):
            return self._scalevaramp(gain) 
        return self.clone(amps=self.amps*gain)

    def oversampled(self, n):
        # type: (int) -> Partial
        dt = self.duration / (self.numbreakpoints*n)
        return self.resampled(dt)

    def fade(self, time, kind='fadein', shape='linear', oversample=4):
        # type: (float, str, str, int) -> Partial
        """
        time: fade time
        kind: one of 'fadein', 'fadeout', 'both'
        shape: the fade shape as string definition ('linear', 'expon(2)', etc)
        """
        assert kind in ("fadein", "fadeout", "both")
        assert time < self.duration
        if kind == 'fadein':
            faded = self.crop(self.t0, self.t0+time)
            curve = bpf.util.makebpf(shape, [faded.t0, faded.t1], [0, 1])
            if shape != 'linear':
                faded = faded.oversampled(oversample)
            faded = faded.scaleamp(curve.map)
            rest = self.crop(self.t0+time+0.001, self.t1)
            return concat(faded, rest)
        elif kind == 'fadeout':
            faded = self.crop(self.t1-time, self.t1)
            curve = bpf.util.makebpf(shape, [faded.t0, faded.t1], [1, 0])
            if shape != 'linear':
                faded = faded.oversampled(oversample)
            faded = faded.scaleamp(curve.map)
            rest = self.crop(self.t0, self.t1-time-0.001)
            return concat(rest, faded)
        elif kind == 'both':
            return self.fade(time, "fadein", shape).fade(time, "fadeout", shape)
        else:
            raise ValueError("kind should be 'fadein', 'fadeout' or 'both'")

    def simplefade(self, fadetime):
        # type: (float) -> Partial
        """
        add a 0 amplitude breakpoint, extending this Partial by given fadetime

        * This is usefull to avoid clipping when manipulating partials.
        * For more complex fades, use Partial.fade
        * If fadetime==0, the amplitude of the edge breakpoints is set to 0
          if it is not already 0

        :param fadetime: float. fadetime for both fadein / fadeout
        :return: Partial
        """
        data = self.toarray()
        t0, f0 = data[0][:2]
        t1, f1 = data[-1][:2]
        if fadetime <= 0:
            raise ValueError("A fadetime must be > 0")
        elif t0-fadetime < 0:
            # would it result in a negative fadetime? Fade from t=0
            data[0, 0] = 0
            data[0, 2] = 0
            if data[1, 0] == 0:
                data[1, 0] = 0.0000000001
                if len(data) > 2:
                    assert data[1, 0] < data[2, 0]
            breakpoint1 = np.array([[t1+fadetime, f1, 0, 0, 0]])
            data = np.vstack((data, breakpoint1))
        else:
            breakpoint0 = np.array([[t0-fadetime, f0, 0, 0, 0]])
            breakpoint1 = np.array([[t1+fadetime, f1, 0, 0, 0]])
            data = np.vstack((breakpoint0, data, breakpoint1))
        return Partial.fromarray(data, label=self.label)
        
    def timewarp(self, curve):
        # type: (Bpf) -> Partial
        """
        Warp the time of this Partial according to the curve.
        curve maps time -> time

        :param curve: Bpf. mapping the time of in this partial to new times,
               possibly in a non-linear way
        :return: Partial
        """
        if self.numbreakpoints <= 2:
            times1 = [curve(t) for t in self.times]
        else:
            times0 = self.times
            times1 = curve.map(times0)
        return self.clone(times=times1)
                
    def freqwarp(self, curve):
        # type: (Bpf) -> Partial
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

    def freqwarp_dynamic(self, gradient):
        # type: (Callable[[float, float], float]) -> Partial
        """
        Similar to freqwarp, but the warping curve can change over time,
        so it takes a gradient (a function of time and freq)

        :param gradient: a Gradient mapping (time, freq) -> freq
        """
        freqs2 = np.fromiter((gradient(t, f) for t, f in zip(self.times, self.freqs)), dtype=float)
        return self.clone(freqs=freqs2)

    def equalize(self, curve):
        # type: (Bpf) -> Partial
        """
        Apply a frequency dependent gain to each breakpoint

        :param curve: maps freq to gain
        :return: Partial
        """
        amps = self.amps * curve.map(self.freqs)
        return self.clone(amps=amps)

    def normalized(self, maxpeak=1.0):
        # type: (float) -> Partial
        """
        Return a normalized version of this Partial

        Example:

        normalized_partial = partial.normalized(db2amp(-3))
        """
        X, Y = self.amp.points()
        gain = maxpeak / np.absolute(Y).max()
        return self.scaleamp(gain)
        
    def resampled(self, dt):
        # type: (float) -> Partial
        """
        Return a new partial resampled at the given time interval

        :param dt: the sampling interval
        :return: the resampled Partial
        """
        # N = (self.t1 - self.t0) / dt
        def snap(x:float, tick:float) -> float:
            return int(x/tick)*tick
        t0 = snap(self.t0, dt)
        t1 = max(snap(self.t1, dt), t0+dt)
        times = np.arange(t0, t1+dt/2, dt)
        if len(times) > 2:
            assert (len(times)-1)/(t1-t0) <= (1/dt)+1, f"t0: {t0} t1: {t1} dt:{dt} times:{times}"
        assert len(times) >= 2
        freqs = self.freq.map(times)
        amps = self.amp.map(times)
        phases = self.phase.map(times) if self.hasphase() else None
        bws = self.bw.map(times) if self.hasbandwidth() else None
        return Partial(times, freqs, amps, phases, bws, label=self.label)

    def shifted(self, dt=0.0, df=0.0):
        # type: (float, float) -> Partial
        """
        Shift this partial on either the time or freq coord

        If partial was defined between time=(1, 2) and freq=(1000, 2000) then

        partial.shifted(0.5, 100) will generate a partial defined 
        within (1.5, 2.5) and (1100, 2100)

        :param dt: delta time
        :param df: delta freq
        :return: Partial
        """
        if dt == 0 and df == 0:
            return self
        if self.t0 + dt < 0:
            raise ValueError("a Partial can't have a negative time")
        data = self.getdata()
        if isinstance(df, _Number) and df != 0:
            freqs = data.freqs + df  # type: np.ndarray
            if (freqs < 0).any():
                raise ValueError("a Partial can't have negative frequencies")
        elif callable(df):
            freqs = bpf.asbpf(df).map(data.times)
            if (freqs < 0).any():
                raise ValueError("a Partial can't have negative frequencies")
        else:
            freqs = data.freqs
        times = data.times + dt if dt != 0 else data.times  # type: np.ndarray
        return Partial(times, freqs, data.amps, data.phases, data.bws, label=self.label)
        
    def quantized(self,
                  pitch_grid=None,  # type: Union[float, np.ndarray, None]
                  db_grid=None,     # type: Union[float, np.ndarray, None]
                  time_grid=None    # type: Union[float, np.ndarray, None]
                  ):
        # type: (...) -> Partial
        """
        Returns a new partial

        pitch_grid: the resolution of the pitch grid, or a seq. of possible pitches
                    (fractional midinotes)
        db_grid: the resolution of the amp grid, in dB, or a list of possible amps,
                 in dB. Use None to not quantize amps
        time_grid: the partial is resampled at the given interval
        
        * NB1: call .simplified to remove redundant breakpoints after quantization
        * NB2: to generate a list of dB as expected in db_grid from a dynamic curve 
               call dyncurve.todbs()
        """
        if time_grid is not None and time_grid > 0:
            data = self.resampled(time_grid).toarray()  # type: np.ndarray
        else:
            data = self.toarray().copy()  # type: np.ndarray
        if pitch_grid is not None:
            if isinstance(pitch_grid, (int, float)):
                pitch_grid_seq = np.arange(0, 127, pitch_grid)
            elif isinstance(pitch_grid, (np.ndarray, list, tuple)):
                pitch_grid_seq = pitch_grid
            else:
                raise TypeError("pitch_grid has an unknown type: {}".format(pitch_grid.__class__))
            midis = [nearest_element(midi, pitch_grid_seq) for midi in map(f2m, data[:, 1])]
            data[:,1] = list(map(m2f, midis))
        if db_grid is not None:
            if isinstance(db_grid, (int, float)):
                db_grid_seq = np.arange(-120, 20, db_grid)
            elif isinstance(db_grid, (np.ndarray, list, tuple)):
                db_grid_seq = db_grid
            else:
                raise TypeError("db_grid needs to be a seq. or a float")
            amp_grid = list(map(db2amp, db_grid_seq))
            data[:,2] = [nearest_element(amp, amp_grid) for amp in data[:, 2]]
        return Partial.fromarray(data, label=self.label)

    def simplify(self, pitchdelta=0.01, dbdelta=0.1, bwdelta=0.01):
        # type: (float, float, float) -> Partial
        """
        Remove breakpoints where the variation is under the given deltas

        Returns a new Partial
        """
        T, F, A, P, B = self.getdata()
        if B is None:
            B = np.zeros_like(T)
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
                if _breakpoint_deviates(b0, b1, b2, dbdelta, pitchdelta, bwdelta):
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


def concat(*partials):
    # type: (*Partial) -> Partial
    """
    Concatenate multiple Partials to produce a new one.
    Assumes that the partials are non-overlapping and sorted
    """
    for p0, p1 in pairwise(partials):
        if p0.t1 > p1.t0:
            raise ValueError("Partials are overlapping, cannot concatenate")
    times = np.concatenate([p.times for p in partials])
    freqs = np.concatenate([p.freqs for p in partials])
    amps = np.concatenate([p.amps for p in partials])
    return Partial(times, freqs, amps)


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
