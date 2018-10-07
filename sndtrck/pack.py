from .partial import concat as concat_partials
from .spectrum import Spectrum
from numpyx import array_is_sorted
import numpy as np
from loristrck import partials_sample, matrix_save as matrix_save_as_sndfile
import logging


logger = logging.getLogger("sndtrck")
INF = float("inf")

def _best_track(tracks, partial, clearance, acceptabledist):
    # type: (List[Partial], Partial, float, flat) -> List[Partial]
    pt0 = partial.t0
    mindist = INF
    best = None
    for track in tracks:
        t1 = track[-1].t1 + clearance
        if t1 < pt0:
            dist = t1 - pt0
            if dist < acceptabledist:
                return track
            if dist < mindist:
                best = track
                mindist = dist
    return best


def _join_track(partials, fade):
    assert fade > 0
    p = concat_partials(partials, fade=fade)
    if p.times[0] > 0:
        assert p.amps[0] == 0
    assert p.amps[-1] == 0
    assert array_is_sorted(p.times)
    return p


def _pack_in_tracks_loris(sp, gap=0.010, fade=0.005):
    # type: (Spectrum, float, float) -> Spectrum
    """
    This is the algorithm used in loris for collation. It is efficient
    but does not render the optimum result. Left here as reference
    """
    tracks = []
    clearance = 2 * fade + gap 
    partials = sorted(sp, key=lambda p:p.t1)
    for p in partials:
        track = next((tr for tr in tracks if tr[-1].t1 + clearance < p.t0), None)
        if track:
            track.append(p)
        else:
            tracks.append([p])
    partials = [_join_track(track, fade=fade) for track in tracks]
    return Spectrum(partials, packed=True)
        

def pack_in_tracks(sp, numtracks=0, gap=0.010, fade=0.005, acceptabledist=0.100):
    # type: (Spectrum, int, float, float) -> Spectrum
    """
    Make a new Spectrum where each Partial is a track. 
    Non-simulateneous partials are packed together into
    tracks, the silence between them is represented as 
    amp=0

    If the spectrum is to be resampled afterwords, gap > samplingPeriod*2

    numtracks: if specified, it sets the number of tracks
               Partials not fitting in will be discarded
               Unused tracks will be discarded
    """
    tracks = []
    clearance = gap + 2*fade
    for partial in sp.partials:
        best_track = _best_track(tracks, partial, clearance, acceptabledist)
        if best_track is not None:
            best_track.append(partial)
        else:
            if numtracks == 0 or len(tracks) < numtracks:
                track = [partial]
                tracks.append(track)

    partials = [_join_track(track, fade=fade) for track in tracks]
    return Spectrum(partials, packed=True)


def _sample_spectrum_interleave_native(sp:Spectrum, dt=0.002) -> np.ndarray:
    t0 = sp.t0
    t1 = sp.t1
    times = np.arange(t0, t1+dt, dt)
    columns = []
    for p in sp:
        columns.append(p.freq.map(times))
        columns.append(p.amp.map(times))
        columns.append(p.bw.map(times))
    bigmatrix = np.column_stack(columns)
    return bigmatrix


def sample_spectrum_interleave(sp:Spectrum, dt=0.002, t0=-1, t1=-1, 
                               maxstreams=0) -> np.ndarray:
    """
    Sample the Partials in sp with a period of dt

    sp: a Spectrum
    dt: the sampling period
    t0: start of sampling time
    t1: end of sampling time
    maxstreams: max. number of simulateous spectral components. When there are more
                components than streams available, the faintest streams are zeroed
                so that they are skipped during resynthesis.

    Assume that `sp` is a Spectrum or a list of Partials where
    each Partial represents a `track` as returned by pack_in_tracks

    That means that each track is itself a Partial resulting of merging
    multiple non-overlapping partials. These tracks are sampled at 
    period `dt` and converted into a bigmatrix with the form:

    [
        [P0.freq@t0, P0.amp@t0, P0.bw@t0, P1.freq@t0, P1.amp@t1, P1.bw@t1, ...],
        [P0.freq@t1, P0.amp@t1, P0.bw@t1, ... ],
        ...
    ]
    
    NB: phase is discarded and partials are resampled at a regular interval.

    The purpose of this routine is to use the data for resynthesis by
    another program (csound, for instance)

    See also: sample_spectrum
    """
    if t0 < 0:
        t0 = sp.t0
    if t1 < 0:
        t1 = sp.t1
    return partials_sample((p.toarray() for p in sp), 
                           dt=dt, t0=t0, t1=t1, maxactive=maxstreams, interleave=True)


def sample_spectrum(sp, dt=0.002, t0=-1, t1=-1, maxstreams=0):
    # type: (Spectrum, float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    Similar to sample_spectrum_interleave, but returns three arrays,
    
    freqs, amps, bws

    In each array, the columns represent each partial, the rows are the
    value for each time.
    """
    if t0 < 0:
        t0 = sp.t0
    if t1 < 0:
        t1 = sp.t1
    return partials_sample((p.toarray() for p in sp),
                           dt=dt, t0=t0, t1=t1, maxactive=maxstreams, interleave=False)
    

def save_spectrum_as_sndfile(sp, outfile, dt=0.002, bits=32):
    """
    Packed spectrum `sp` is sampled at period `dt` and the resulting matrix
    is saved to `outfile` as a mono 32-bit float (or 64 bits, if specified) 
    PCM soundfile. The resulting soundfile has the following format:

    dataOffset=5, dt, numcols, numrows, t0, *data

    where data is the matrix returned by `sample_spectrum_interleave`:

    dataOffset: the offset into the file (as number of samples) where
                the data begins. In this case, dataOffset equals 5
    dt: sampling period
    numcols, numrows: shape of the matrix
    t0: beginning of sampling time. t0 = sp.t0
    bits: 32 or 64

    NB: if `sp` was not packed before, it is packed with sensible
        defaults. 

    """
    if not sp.packed:
        logger.warn("Spectrum not packed. Packing with default values")
        sp = pack_in_tracks(sp, gap=dt*3, fade=0.005) 
    m = sample_spectrum_interleave(sp, dt)
    matrix_save_as_sndfile(m, outfile, dt, sp.t0, bits=bits)
    return m
