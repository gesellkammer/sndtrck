from __future__ import absolute_import
import bpf4
import numpy as np
from . import spectrum

Bpf = bpf4.BpfInterface


def bpf2partial(freq, amp, dt=0.010):
    # type: (Bpf, Bpf, float) -> spectrum.Partial
    """
    Create a Partial from a bpf representing
    frequency and a number or bpf representing amplitude

    freq: a bpf representing a frequency curve
    amp: a bpf representing an amplitude envelope
    dt: the sampling period to sample the curves. Resampling is done if dt > 0

    Example:

    import bpf4 as bpf
    freq = bpf.linear(0, 60, 10, 72).m2f()
    partial = bpf2partial(freq, bpf.asbpf(0.5))
    Spectrum([partial]).show()
    """
    f = bpf4.asbpf(freq)
    a = bpf4.asbpf(amp)
    x0 = max(f.bounds()[0], a.bounds()[0])
    x1 = min(f.bounds()[1], a.bounds()[1])
    if dt <= 0:
        raise ValueError("The sampling interval `dt` should be > 0")
    numsamples = int((x1-x0)/dt)
    times = np.linspace(x0, x1, numsamples)
    freqs = f.mapn_between(numsamples, x0, x1)
    amps = a.map(times)
    assert len(times) == len(freqs) == len(amps)
    p = spectrum.Partial(times, freqs, amps)
    return p
