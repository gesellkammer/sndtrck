import functools
from . import io
from . import spectrum


@functools.wraps(io.analyze)
def analyze(*args, **kws):
    partialdata = io.analyze(*args, **kws)
    return spectrum.fromarray(partialdata)
