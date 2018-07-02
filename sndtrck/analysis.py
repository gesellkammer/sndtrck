import functools
from . import io
from . import spectrum
from . import backend_loris


@functools.wraps(backend_loris.analyze)
def analyze(*args, **kws):
    partialdata = backend_loris.analyze(*args, **kws)
    return spectrum.fromarray(partialdata)


@functools.wraps(backend_loris.analyze_samples)
def analyze_samples(*args, **kws):
    partialdata = backend_loris.analyze_samples(*args, **kws)
    return spectrum.fromarray(partialdata)
