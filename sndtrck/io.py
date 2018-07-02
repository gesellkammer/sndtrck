import warnings
import subprocess as _subprocess
import platform as _platform
import numpy as np
from collections import namedtuple
import logging

from . import backend_loris
from . import backend_pysdif

from .errors import *
from .util import *
import bpf4 as bpf
from typing import Iterable as I, Optional as Opt, Tuple as Tup, List
from .backend_loris import analyze as analyze

BPF = bpf.BpfInterface

"""
IO

This is a 'floor' module, in-out is done via built-in datatypes

Conversions to Spectrum are done in spectrum.py

NB: we implement our conversions here because we don't need to
import spectrum for that, we convert a Spectrum to s array
and go with that. For that reason the to* functions can be
exported as is, the from* functions need to be wrapped one
level higher

For soundfile io, use sndfileio directly
"""

logger = logging.getLogger("sndtrck")


def tosdif(matrices, labels, outfile, rbep=True, fadetime=0):
    # type: (I[np.ndarray], I[int], str, bool, float) -> None
    """
    Write this spectrum as SDIF

    rbep: if True, use the given timing, do not resample to a common-grid
    fadetime: if > 0, add a fadein-out to the partials when the first or
              last breakpoint has an amplitude > 0
    """
    outfile = normalizepath(outfile)
    # backends = [backend_pysdif, backend_loris]
    backends = [backend_loris]
    backends = [b for b in backends if b.is_available() and b.get_info()['write_sdif']]
    if not backends:
        raise FunctionalityNotAvailable("no backend implements writing to sdif")
    backend = backends[0]
    return backend.write_sdif(outfile, matrices, labels=labels, rbep=rbep, fadetime=fadetime)


def fromsdif(sdiffile):
    # type: (str) -> Tup[List[np.ndarray], List[int]]
    """
    Reads a SDIF file (1TRC or RBEP)

    Returns a tuple (matrices, labels) as expected by fromarray

    Raises FunctionalityNotAvailable if no backend implements this
    """
    if not os.path.exists(sdiffile):
        raise OSError("SDIF file not found!")
    from . import backend_loris
    backends = [backend_loris]
    for backend in backends:
        if backend.is_available() and backend.get_info()['read_sdif']:
            return backend.read_sdif(sdiffile)
    raise FunctionalityNotAvailable("No Backend implements SDIF reading")


def fromtxt(path):
    # type: (str) -> Tup[List[np.ndarray], List[int]]
    """
    Read a Spectrum from a txt file in the format used by SPEAR
    Returns a tuple (matrices, labels) to be passed tp fromarray
    """
    f = open(path)
    it = iter(f)
    next(it)
    next(it)
    npartials = int(next(it).split()[1])
    next(it)
    EPSILON = 1e-10
    skipped = 0
    matrices = []  # type: List[np.ndarray]
    labels = []    # tyoe: List[int]
    while npartials > 0:
        partial_id = int(float(next(it).split()[0]))  # type: int
        data = np.fromstring(next(it), sep=" ", dtype=float)
        times = data[::3]    # type: np.ndarray
        freqs = data[1::3]   # type: np.ndarray
        amps = data[2::3]    # type: np.ndarray
        # check if any point has the same time as the previous one
        # if this is the case, shift the second point to the right
        # a minimal amount
        if len(times) > 2:
            for _ in range(10):
                same = times[1:] - times[:-1] == 0   # type: np.ndarray
                if same.any():
                    logger.warning("duplicate points found")
                    times[1:][same] += EPSILON
                else:
                    break
        dur = times[-1] - times[0]
        if dur > 0:
            bws = phases = np.zeros_like(amps)
            partial = np.column_stack((times, freqs, amps, phases, bws))
            matrices.append(partial)
            labels.append(partial_id)
        else:
            skipped += 1
        npartials -= 1
    if skipped:
        logger.warning("Skipped %d partials without duration" % skipped)
    return matrices, labels


def tospear(matrices, outfile, use_comma=False):
    # type: (I[np.ndarray], str, bool) -> None
    """
    writes the partials in the text format defined by SPEAR
    (Export Format/Text - Partials)

    The IDs of the partials are lost, partials are enumerated
    in the order defined in the Spectrum
    """
    arraylist = aslist(matrices)
    outfile = normalizepath(outfile)
    f = open(outfile, 'w', encoding="ascii")
    f_write = f.write
    arraylist = aslist(arraylist)
    f_write("par-text-partials-format\n")
    f_write("point-type time frequency amplitude\n")
    f_write("partials-count %d\n"%len(arraylist))
    f_write("partials-data\n")
    for i, m in enumerate(arraylist):
        # times, freqs = p.freq.points()
        # _, amps = p.amp.points()
        # data = column_stack((times, freqs, amps)).flatten()
        data = m[:, 0:3].flatten()
        t0 = data[0, 0]
        t1 = data[-1, 0]
        header = "%d %d %f %f\n" % (i, len(m), t0, t1)
        # datastr = " ".join(b"%f" % n for n in data)
        datastr=" ".join("%f" % n for n in data)
        if use_comma:
            header = header.replace(".", ",")
            datastr = datastr.replace(".", ",")
        f_write(header)
        f_write(datastr)
        f_write('\n')


def tonpz(matrices, labels, outfile):
    # type: (I[np.ndarray], List[int], str) -> None
    """
    write the spectrum as a numpy .npz file
    """
    # numchars = len(str(len(spectrum)))
    # keyformat = "partial_%.{numchars}d".format(numchars=numchars)
    
    def dict2array(d):
        return np.array(list(zip(d.keys(), d.values())))

    keyformat = "_%d"
    _metadata = {
        b"version": b"1.0",
        b"numpartials": str(len(labels)).encode("ascii"),
        b"columns": b"time,freq,amp,phase,bw",
        b"partialprefix": b"_"
    }
    metadata = dict2array(_metadata)
    partials = {keyformat%i: matrix for i, matrix in enumerate(matrices)}
    np.savez(outfile, metadata=metadata, labels=labels, **partials)


def fromnpz(path):
    # type: (str) -> Tup[List[np.ndarray], List[int]]
    npz = np.load(path)
    labels = npz['labels']
    metadata = npz['metadata']
    metadata = dict(metadata)
    version = metadata[b'version'].decode("ascii").split(".")
    if version[0] == "1":
        matrices = [None] * int(metadata[b'numpartials'])  # type: List[Opt[np.ndarray]]
        partialprefix = metadata[b'partialprefix'].decode("ascii")
        skip = len(partialprefix)
        for key, value in npz.items():
            if key.startswith(partialprefix):
                idx = int(key[skip:])
                matrices[idx] = value
    else:
        raise ValueError("Version not recognized")
    # assert all(m is not None for m in matrices)
    return matrices, labels


# noinspection PyUnresolvedReferences
def tohdf5(matrices, labels, outfile):
    # type: (I[np.ndarray], I[int], str) -> None
    """
    save to outfile as HDF5

    format:
        / --> attributes:
                version: numeric version of the format
                numpartials: number of partials
                columns: name of the columns
            labels: array of the numeric labels of each partial
            partials/
                0
                1
                2
                ...
                N

    where each partial is a matrix with the given columns

    """
    try:
        import h5py
    except ImportError:
        raise ImportError("can't use h5 backend, h5py not available")

    f = h5py.File(outfile, "w", libver='latest')
    columns = "time freq amp phase bw"
    labellist = aslist(labels)  # type: List[int]
    metadata = {
        'version': 1.1,
        'numpartials': len(labellist),
        'columns': columns.split()
    }
    for key, value in metadata.items():
        f.attrs.create(key, data=value)
    f.create_dataset("labels", data=labellist)
    partialroot = f.create_group("partials")
    numchars = len("999999")
    keyformat = "%.{numchars}d".format(numchars=numchars)
    for i, matrix in enumerate(matrices):
        key = keyformat % i
        partialroot.create_dataset(key, data=matrix)
    f.flush()
    f.close()


# noinspection PyUnresolvedReferences
def fromhdf5(path):
    # type: (str) -> Tup[List[np.ndarray], List[int]]
    """
    Reads a spectrum from a HDF5 file.
    Returns a tuple (partials, labels)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("Could not find h5py. HDF5 support is not available")
    store = h5py.File(path)
    # check version
    version = float(store.attrs.get('version', 1.0))
    # numpartials = store.attrs.get('numpartials')
    if not (1.0 <= version < 2.0):
        warnings.warn("Version not supported: %s" % str(version))
    # Version 1.0
    saved_labels = store.get('labels', None)
    matrices = []   # type: List[np.ndarray]
    labels = []     # type: List[int]
    for matrix in store.get('partials').values():
        a = matrix.value
        times = a[:, 0]
        freqs = a[:, 1]
        amps = a[:, 2]
        if a.shape[1] == 3:
            phases = bws = np.zeros_like(amps)
        else:
            phases = a[:, 3]
            bws = a[:, 4]
        partial_index = int(matrix.name.split("/")[-1])
        try:
            label = saved_labels[partial_index] if labels else 0
        except IndexError:
            label = 0
        labels.append(label)
        partial = np.column_stack((times, freqs, amps, phases, bws))
        matrices.append(partial)
    store.close()
    return matrices, labels

def synthesize(matrices, samplerate=44100):
    # type: (I[np.ndarray], int) -> np.ndarray
    """
    Synthesize a Spectrum

    Returns: a numpy array with the samples

    NB: to synthesize part of the Spectrum, call
        spectrum.partials_between(start, end, crop=True)
    """
    logger.warning("Deprecated, use synthesis.synthesize directly")

    for backend in [backend_loris]:
        if backend.is_available() and backend.get_info().get('synthesize', False):
            return backend.synthesize(matrices, samplerate)
    raise BackendNotAvailable("No backend can synthesize this spectrum")


EstimateF0 = namedtuple("EstimateF0", ["freq", "confidence"])


def estimatef0(matrices, minfreq=30, maxfreq=3000, interval=0.1,
               minconfidence=None):
    # type: (I[np.ndarray], float, float, float, bool) -> EstimateF0
    """
    Estimate the fundamental of this spectrum

    * spectrum: a Spectrum
    * minfreq, maxfreq: the frequency range where a fundamental is expected
    * interval: the time accuracy of the measurement
    * minconfidence: if set, freq will be 0 whenever the confidence is below
                     this value. In this way you can check the confidence just
                     by checking if freq. > 0 freq will always be 0 whenever
                     there is absolute silence in the spectrum

    Returns: (freq, confidence), where

    - freq: a bpf representing the frequency of the fundamental at time `t`
    - coef: a bpf representing the confidence of the measurement at time `t`
            Values of 0.9-1 represent high confidence, 0 represents no partial
            at time `t`, values in between represent unvoiced sounds
    """
    backends = [backend_loris]
    backends_ok = [b for b in backends if
                   b.is_available() and b.get_info().get('estimatef0', False)]
    if not backends_ok:
        raise FunctionalityNotAvailable("No backend implements `estimatef0`")
    freq, conf = backends[0].estimatef0(matrices, minfreq, maxfreq, interval)
    if minconfidence is not None:
        freq *= (conf >= minconfidence)
    return EstimateF0(freq, conf)

    
# -------
# HELPERS
# ------- 


def open_spectrum_in_spear(filepath, wait=True):
    # type: (str, bool) -> None
    def osx(path):
        if not wait:
            raise NotImplemented("waiting is not implemented in OSX")
        _subprocess.call('open -a SPEAR "%s"' % path, shell=True)

    def win(path):
        raise NotImplementedError()

    def linux(path):
        locations = ["~/.wine/drive_c/Program Files (x86)/SPEAR/spear.exe"]
        for location in locations:
            spearexe = os.path.abspath(os.path.expanduser(location))
            if os.path.exists(spearexe):
                break
        else:
            raise RuntimeError("SPEAR was not found. Locations searched: %s"
                               % str(locations))
        path = os.path.abspath(path)
        cmd = 'wine "%s" %s' % (spearexe, path)
        print(cmd)
        if wait:
            return _subprocess.call(cmd, shell=True)
        else:
            return _subprocess.Popen(cmd, shell=True)

    f = {
        'Darwin': osx,
        'Windows': win,
        'Linux': linux
    }.get(_platform.system())
    if f:
        f(filepath)

