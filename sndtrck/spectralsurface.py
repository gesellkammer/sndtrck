from .spectrum import Spectrum
import bisect
from emlib.pitchtools import amp2db, db2amp
from emlib.interpol import interpol_linear as _linlin
from typing import Sequence as _S


class SpectralSurface:

    def __init__(self, sp:Spectrum, decay=2, mindb=-120):
        """
        decay: dB/Hz. decay=2 --> in 30 Hz the amplitude decays 60 dB
        interp: one of 'linear', 'expon(exp)'
        """
        self.sp = sp
        self.decay = decay
        self.mindb = mindb
        # self.interp = interp

    def nearest(self, time:float, freq:float, timemargin=0.01) -> float:
        s = self.sp.partials_between(time-timemargin, time+timemargin)
        if not s:
            return (0, 0)
        mindist = float("inf")
        for p in s:
            p_freq = p.freq(time)
            dist = abs(p_freq - freq)
            if dist < mindist:
                mindist = dist
                best_freq = p_freq
                best_partial = p
        best_amp = best_partial.amp(time)
        db = max(amp2db(best_amp) - mindist*self.decay, self.mindb)
        return best_freq, db2amp(db)

    def nearestx(self, time:float, freqs:_S[float], timemargin=0.01) -> _S[float]:
        """
        Return a list of (freq, amp), representing the 
        freq and amp of the nearest Partial for each given freq.

        Example:

        freqs = [100, 200]
        out = surface.nearest(0.5, freqs)
        for freq, row in zip(freq, out):
            print(f"Nearest partial from {freq}Hz @ 0.5s has a freq. of {row[0]}")

        """
        s = self.sp.partials_between(time-timemargin, time+timemargin)
        if not s:
            return [(0, 0)] * len(freqs)
        data = [(p.freq(time), p.amp(time)) for p in s]
        mindb = self.mindb
        decay = self.decay
        out = []
        for freq in freqs:
            nearest = min(data, key=lambda row: abs(row[0] - freq))
            f, a = nearest
            db = amp2db(a)
            diff = abs(f - freq)
            db2 = max(db - diff * decay, mindb)
            out.append((f, db2amp(db2)))
        return out

    def at(self, time:float, freq:float) -> float:
        return self.atx(time, [freq])[0]

    def atx(self, time:float, freqs:_S[float]) -> _S[float]:
        s = self.sp.partials_at(time)
        data = [(p.freq(time), p.amp(time)) for p in s]
        data.sort()
        out = []
        d = self.decay
        mindb = self.mindb
        for freq in freqs:
            idx1 = bisect.bisect(data, (freq, 0))
            if idx1 >= len(data):
                idx1 = idx0 = idx1 - 1
            idx0 = max(0, idx1-1)
            f0, a0 = data[idx0]
            f1, a1 = data[idx1]
            a0db = amp2db(a0)
            a1db = amp2db(a1)
            df0 = (a0db - mindb) / d
            df1 = (a1db - mindb) / d
            if f1-df1 <= freq <= f0+df0:
                db0 = _linlin(freq, f0, f0+df0, a0db, -120)
                db1 = _linlin(freq, f1-df1, f1, -120, a1db)
                db = max(db0, db1)   
            elif freq <= f0+df0:
                db = _linlin(freq, f0, f0+df0, a0db, -120)
            elif f1 - df1 <= freq:
                db = _linlin(freq, f1-df1, f1, -120, a1db)
            else:
                db = mindb
            out.append(db2amp(db))
        return out
        """
        df = 10000
        l1 = bpf.linear(
            f0-df, a0db - d*df, 
            f0, a0db,
            f0+df, a0db - d*df
        )
        l2 = bpf.linear(
            f1-df, a1db-d*df,
            f1, a1db,
            f1+df, a1db-d*df
        )
        l3 = bpf.max_(l1, l2, bpf.const(self.mindb))
        amp = db2amp(l3(freq))
        return amp
        """
