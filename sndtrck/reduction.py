from math import sqrt
import bpf4 as bpf

from emlib.pitchtools import db2amp

from .spectrum import Spectrum
from .partial import Partial
from .typehints import List, Tup, U


class PartialWeighter:
    def __init__(self, freqcurve, durcurve, ampcurve, freqweight=1, durweight=1, ampweight=1):
        self.freqcurve = freqcurve
        self.durcurve = durcurve
        self.ampcurve = ampcurve
        self.freqweight = freqweight
        self.durweight = durweight
        self.ampweight = ampweight

    def factor(self, partial: Partial):
        freqfactor = self.freqcurve(partial.meanfreq_weighted)
        durfactor = self.durcurve(partial.duration)
        ampfactor = self.ampcurve(partial.meanamp)
        return sqrt(freqfactor**2 * self.freqweight + 
                    durfactor**2 * self.durweight + 
                    ampfactor**2 * self.ampweight)

    def rate(self, partial: Partial):
        return partial.energy() * self.factor(partial)

    def select(self, partials: List[Partial], n: int) -> Tup[List[Partial], List[Partial]]:
        return _select_best(partials, n, self)


def make_weighter(mode='speech'):
    if mode == 'speech':
        freqcurve = bpf.linear(0, 0, 50, 0, 80, 1, 200, 1, 250, 2, 4500, 2, 6000, 1, 8000, 1, 10000, 0)
        durcurve = bpf.linear(0, 0, 0.01, 0, 0.02, 1)
        ampcurve = bpf.linear(0, 0, db2amp(-75), 0, db2amp(-65), 1)
    elif mode == 'transcription':
        freqcurve = bpf.linear(0, 0, 30, 0, 50, 1, 4500, 1, 5000, 0)
        durcurve = bpf.linear(0, 0, 0.01, 0, 0.02, 1)
        ampcurve = bpf.linear(0, 0, db2amp(-70), 0, db2amp(-60), 1)
    elif mode == 'default':
        freqcurve = bpf.linear(0, 0, 30, 0, 50, 1, 8000, 1, 12000, 0.5, 16000, 0)
        durcurve = bpf.linear(0, 0, 0.01, 0, 0.02, 1)
        ampcurve = bpf.linear(0, 0, db2amp(-80), 0, db2amp(-60), 1)
    elif mode == 'speech-transcription':
        freqcurve = bpf.linear(0, 0, 50, 0, 80, 1, 200, 1, 250, 2, 4500, 2, 5000, 0)
        durcurve = bpf.linear(0, 0, 0.01, 0, 0.02, 1)
        ampcurve = bpf.linear(0, 0, db2amp(-75), 0, db2amp(-65), 1)
    else:
        raise KeyError("mode unknown, must be one of 'speech', 'transcription'")
    return PartialWeighter(freqcurve=freqcurve, durcurve=durcurve, ampcurve=ampcurve, 
                           freqweight=1, durweight=1, ampweight=2)


def get_default_weighter() -> PartialWeighter:
    return make_weighter(mode='default')
    

def _select_best(partials: List[Partial], n: int, weighter: PartialWeighter) -> Tup[List[Partial], List[Partial]]:
    """
    Given a list of partials, select the n best, based on partial weight
    """
    if len(partials) <= n:
        return partials, []
    sortedpartials = sorted(partials, key=weighter.rate, reverse=True)
    ok = sortedpartials[:n]
    notok = sortedpartials[n:]
    return ok, notok


def adaptive_filter(sp: Spectrum, 
                    max_simultaneous_partials: int, 
                    windowdur: float=0.1, 
                    hopdur: float=None, 
                    weighter:U[str, PartialWeighter]=None, 
                    multipass=False) -> Tup[Spectrum, Spectrum]:
    """
    Args
        sp: the spectrum which needs to be reduced
        max_simultaneous_partials: the maximum number of simultaneous partials
        windowdur: the duration of the window to analyze
        hopdur: the time to skeep between analysis steps
        weighter: a PartialWeighter (see make_weighter)
        multipass: if True, the spectrum is filtered using multiple passes based on the given arguments

    Returns
        A tuple of spectrums (collected, filtered)
    """
    if weighter is None:
        weighter = get_default_weighter()
    elif isinstance(weighter, str):
        weighter = make_weighter(weighter)
    
    assert isinstance(weighter, PartialWeighter)

    if multipass:
        n0 = int(max_simultaneous_partials * 0.38)
        n1 = max_simultaneous_partials - n0
        ws = [max(sp.duration / 20, windowdur*8), windowdur]
        return adaptive_filter_multipass(sp, max_simultaneous_partials=[n0, n1], windowdurs=ws, weighter=weighter)

    if hopdur is None:
        hopdur = windowdur

    t0 = sp.t0
    collected_indices = set()
    for i, p in enumerate(sp):
        p.selection = i

    while t0 < sp.t1:
        t1 = t0 + windowdur
        active = sp.partials_between(t0, t1)
        # active = [p for p in active if p.selection not in collected_indices]
        ok, notok = _select_best(active, max_simultaneous_partials, weighter=weighter)
        for p in ok:
            collected_indices.add(p.selection)
        t0 += hopdur
    collected_partials = [sp[idx] for idx in collected_indices]
    collected = Spectrum(collected_partials)
    filtered_partials = [p for p in sp if p.selection not in collected_indices]
    filtered = Spectrum(filtered_partials)
    return collected, filtered 


def adaptive_filter_multipass(sp: Spectrum, max_simultaneous_partials: List[int], windowdurs: List[int], 
                              weighter: PartialWeighter=None) -> Tup[Spectrum, Spectrum]:
    """
    Similar to adaptive_filter, but multiple passes are performed

    max_simultaneous_partials: a list with the value for each pass
    windowdurs: a list with the value for each pass
    """
    selected = []
    for n, windowdur in zip(max_simultaneous_partials, windowdurs):
        ok, notok = adaptive_filter(sp, max_simultaneous_partials=n, windowdur=windowdur, weighter=weighter, multipass=False)
        selected.extend(ok.partials)
        sp = notok
    return Spectrum(selected), Spectrum(notok.partials)
