from __future__ import absolute_import, annotations
from emlib.pitchtools import db2amp, m2f, f2m
import bpf4 as bpf
from . import spectrum as sp
from . import typehints as t


Bpf = bpf.BpfInterface


def contrast(spectrum: sp.Spectrum, mid:float, exp=1.0) -> sp.Spectrum:
    """
    Change the contrast of spectrum

    mid: a dB amplitude
    exp: 0 - no effect
         1 - full effect 
         > 1: possible, needs rescaling

    formula: newamps = amps * (amps/mid)**exp
    """ 
    assert mid <= 0
    mid = db2amp(mid)
    newpartials = []
    for partial in spectrum.partials:
        A = partial.amps
        B = A * (A/mid)**exp
        newpartial = partial.clone(amps=B)
        newpartials.append(newpartial)
    return spectrum.__class__(newpartials)


def normalize(spectrum: sp.Spectrum, maxpeak=1.0) -> sp.Spectrum:
    maxamp = max(p.amps.max() for p in spectrum.partials)
    ratio = maxpeak / maxamp
    partials = [p.scaleamp(ratio) for p in spectrum.partials]
    return sp.Spectrum(partials)


def transpose(spectrum: sp.Spectrum, semitones:float) -> sp.Spectrum:
    """
    Transpose spectrum by a fixed number of semitones
    """
    curve = bpf.asbpf(lambda f: m2f((f2m(f)+semitones)))
    return spectrum.freqwarp(curve)


def transpose_dynamic(spectrum: sp.Spectrum, curve) -> sp.Spectrum:
    """
    Transpose a spectrum with a time-varying semitone curve

    curve: function (time: float) -> semitone_transposition: float
    """
    def gradient(t, f):
        semitones = curve(t)
        f2 = m2f(f2m(f) + semitones)
        return f2
    return spectrum.freqwarp_dynamic(gradient)


def sort_by_energy(spectrum: sp.Spectrum) -> t.List[sp.Partial]:
    """
    Returns a list of Partials sorted by descending energy 
    (more energy first)
    """
    out = sorted(spectrum.partials, key=lambda partial:partial.energy(), reverse=True)
    return out


def loudest_partials(spectrum: sp.Spectrum, maxpartials: int) -> sp.Spectrum:
    partials = sort_by_energy(spectrum)
    return sp.Spectrum(partials[:maxpartials])