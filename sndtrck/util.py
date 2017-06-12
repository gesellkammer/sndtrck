from __future__ import absolute_import
import sndfileio as _sndfileio
import os
import numpy as np
from emlib.pitch import f2m, amp2db
import typing as t


def aslist(x):
    # type: (t.Iterable) -> t.List
    if isinstance(x, list):
        return x
    return list(x)


def _interpol_slow(x, x0, y0, x1, y1):
    # type: (float, float, float, float, float) -> float
    return (x-x0)/(x1-x0)*(y1-y0)+y0

try:
    from interpoltools import interpol_linear
except ImportError:
    interpol_linear = _interpol_slow


def isiterable(obj):
    # type: (t.Any) -> bool
    return hasattr(obj, '__iter__') and not isinstance(obj, str)


def normalizepath(path):
    # type: (str) -> str
    return os.path.abspath(os.path.expanduser(path))


def sndreadmono(path, channel=1):
    # type: (str, int) -> t.Tuple[np.ndarray, int]
    """
    Read the soundfile and return a tuple (samples, sr) as float64
    If soundfile is not mono, convert it to mono.
    """
    samples, sr = _sndfileio.sndread(path)
    monosamples = _sndfileio.asmono(samples, channel)
    return monosamples, sr


def sndwrite(samples, samplerate, path):
    # type: (np.ndarray, int, str) -> None
    assert isinstance(samples, np.ndarray)
    assert isinstance(samplerate, int)
    assert isinstance(path, str)
    _sndfileio.sndwrite(samples, samplerate, path)


def array_snap(a, delta):
    # type: (np.ndarray, float) -> np.ndarray
    return np.round(a/delta) * delta