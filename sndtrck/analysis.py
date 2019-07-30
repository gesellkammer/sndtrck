import functools
from . import spectrum
from . import backend_loris as _loris


__all__ = [
    "analyze",
    "analyze_samples"
]



@functools.wraps(_loris.analyze)
def analyze(*args, **kws):
    partialdata = _loris.analyze(*args, **kws)
    return spectrum.fromarray(partialdata)


@functools.wraps(_loris.analyze_samples)
def analyze_samples(*args, **kws):
    partialdata = _loris.analyze_samples(*args, **kws)
    return spectrum.fromarray(partialdata)
