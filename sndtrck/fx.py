from __future__ import absolute_import
# from em.pitchtools import db2amp, m2f, f2m
from emlib.pitch import db2amp, m2f, f2m
import bpf4 as bpf
from . import spectrum as sp
from typing import Callable


Bpf = bpf.BpfInterface


def contrast(spectrum, mid, exp=1):
    # type: (sp.Spectrum, float, float) -> sp.Spectrum
    """
    Change the contrast of spectrum

    mid: a dB amplitude
    exp: 0 - no effect
         1 - full effect 
         > 1: possible, needs rescaling

    formula: B = A * (A/mid)**exp
    """
    assert mid <= 0
    mid = db2amp(mid)
    newpartials = []
    for partial in spectrum.partials:
        A = partial.amp.points()[1]
        B = A * (A/mid)**exp
        newpartial = partial.clone(amps=B)
        newpartials.append(newpartial)
    return spectrum.__class__(newpartials)


def normalize(spectrum, maxpeak=1):
    # type: (sp.Spectrum, float) -> sp.Spectrum
    maxamp = max(p.amps.max() for p in spectrum.partials)
    ratio = maxpeak / maxamp
    partials = [p.scaleamp(ratio) for p in spectrum.partials]
    return spectrum.__class__(partials)


def transpose(spectrum, semitones):
    # type: (sp.Spectrum, float) -> sp.Spectrum
    """
    Transpose spectrum by a fixed number of semitones
    """
    curve = bpf.asbpf(lambda f: m2f((f2m(f)+semitones)))
    return spectrum.freqwarp(curve)


def transpose_dynamic(spectrum, curve):
    # type: (sp.Spectrum, Callable[[float], float]) -> sp.Spectrum
    """
    Transpose a spectrum with a time-varying semitone curve

    curve(t) -> semitone transposition
    """
    def gradient(t, f):
        semitones = curve(t)
        f2 = m2f(f2m(f) + semitones)
        return f2
    return spectrum.freqwarp_dynamic(gradient)

