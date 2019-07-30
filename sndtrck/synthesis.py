import tempfile
import liblo
import time
import atexit
import os
import sys
import logging
import numpy as np
import bpf4
from loristrck import matrix_save as matrix_save_as_sndfile
import subprocess
from functools import lru_cache

from .spectrum import Spectrum
from .background import deferred as _deferred
from .pack import pack_in_tracks, sample_spectrum_interleave
from .partial import Partial
from .config import getconfig
from .util import sndread, sndwrite
from .typehints import Opt


logger = logging.getLogger("sndtrck")

Bpf = bpf4.BpfInterface


def bpf2partial(freq, amp, bw=None, dt=0.010):
    # type: (Bpf, Bpf, float) -> Partial
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
    if bw:
        bws = bpf4.asbpf(bw).map(times)
        
    assert len(times) == len(freqs) == len(amps)
    p = Partial(times, freqs, amps, bws=bws)
    return p


def _get_audio_backend():
    # type: () -> Opt[str]
    """
    Returns a backend for this platform, according
    to the config. 
    """
    config = getconfig()
    platform = sys.platform
    if platform.startswith("linux"):
        backend = config['csound.backend.linux']
    elif platform == "win32":
        backend = config['csound.backend.win']
    elif platform == "darwin":
        backend = config['csound.backend.macos']
    else:
        logger.error(f"_get_audio_backend: Platform not known: {platform}")
        return None
    return backend.strip()


def _find_csound():
    if sys.platform == 'linux' or sys.platform == 'darwin':
        which = subprocess.getoutput("which csound")
        if which:
            return which
        for path in ["/usr/local/bin/csound"]:
            if os.path.exists(path):
                return path
        return None
    else:
        return None


_backends_which_support_systemsr = {'jack', 'alsa', 'coreaudio'}
_backends_which_need_realtime = {'alsa', 'portaudio'}


def _backend_available(backend):
    if backend == 'jack':
        status = int(subprocess.getstatusoutput("jack_control status")[0])
        return status == 0
    else:
        return True

def _get_system_sr(backend=None):
    """
    Query either given backend or the configured backend for
    the current platform

    Returns a tuple (backend, sr)

    If the backend supports a system samplerate, it returns the
    system samplerate, otherwise it returns a possible samplerate
    for that backend

    Config:

    '{platform}.csound.backend'
    """
    # return ('jack', 44100)
    if backend is None:
        backend = _get_audio_backend()
    logger.debug(f"get_system_sr: testing backend: {backend}")
    if backend not in _backends_which_support_systemsr:
        return backend, 44100
    if backend == 'jack':
        if not _backend_available('jack'):
            logger.warn("get_system_sr: jack does not seam to be running")
            sr = None
        else:
            sr = int(subprocess.getoutput("jack_samplerate"))
    else:
        sr = _csound_get_sr(backend)
    return backend, sr

    
def _get_free_oscport():
    s = liblo.Server()
    port = s.port
    s.free()
    return port


def _csound_get_sr(backend):
    args = f"csound -odac -+rtaudio={backend} --get-system-sr".split()
    try:
        proc = subprocess.Popen(args, stderr=subprocess.PIPE, 
                                stdout=subprocess.PIPE)
        proc.wait()
        srlines = [line for line in proc.stdout.readlines() 
                   if line.startswith(b"system sr:")]
        if not srlines:
            logger.error(f"csound_get_sr: Failed to get sr with backend {backend}")
            sr = None
        else:
            sr = float(srlines[0].split(b":")[1].strip())
            logger.debug(f"csound_get_sr: sample rate query output: {srlines}")
            if sr < 0:
                # this happens when csound can't determine the sample rate
                # this is the case when pulseaudio is present, even
                # with alsa as backend
                if backend == 'portaudio':
                    # this should not happen
                    logger.error(f"csound_get_sr: portaudio returned a sr: {sr}!?!")
                    return (backend, 44100)
                _, sr = _get_system_sr("portaudio")
    except FileNotFoundError:
        csoundpath = _find_csound()
        raise FileNotFoundError(f"Could not find csound. Csound path: {csoundpath}")
    return sr
        

def _remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


@lru_cache(maxsize=50)
def pack_in_tracks_cached(sp, gap):
    logger.debug("pack_in_tracks_cached: packing, cache missed")
    return pack_in_tracks(sp, gap=gap)


def _estimate_dt(sp: Spectrum, render=False):
    cfg = getconfig()
    partial_percentile=cfg['breakpointgap.partial_percentile']
    if render:
        percentile = 0
    else:
        percentile = cfg['breakpointgap.percentile']
    gap = sp.estimate_breakpoint_gap(percentile=percentile,
                                     partial_percentile=partial_percentile)
    kr = 64/41000.
    dt = max(gap, kr)
    return round(dt, 4)


@lru_cache(maxsize=50)
def sample_spectrum_interleave_cached(sp, dt):
    return sample_spectrum_interleave(sp, dt)


def _spectrum_to_matrix(sp, wavfile=None, dt=None):
    if dt is None:
        dt = _estimate_dt(sp)
    gap = dt*3
    if not sp.packed:
        sp = pack_in_tracks_cached(sp, gap)
    m = sample_spectrum_interleave_cached(sp, dt)
    if wavfile is None:
        wavfile = tempfile.mktemp(suffix=".wav", prefix="mtx-")
    matrix_save_as_sndfile(m, wavfile, dt, sp.t0)
    return wavfile
    

class SpectrumPlayer:

    def __init__(self, sp, sr=None, sndfile=None, autoplay=False, looping=False,
                 speed=1, start=None, end=None, dt=None,  
                 exitwhendone=False, on_pos=None, on_play=None, 
                 oscfreq=None, interactive=True, gain=1, verbose=None):
        """
        sp: a packed Spectrum (see pack_in_tracks)
        on_pos: an optional callback to be called when the playhead changed
        on_play: an optional callback to be called when the playstate changed
        oscport: if None, a free port will be allocated
        """
        config = getconfig()
        if sndfile is not None:
            if interactive:
                logger.info("SpectrumPlayer: When setting sndfile, rendering will be offline"
                            "and cannot be interactive")
                interactive = False
            if sr is None:
                sr = config['render.samplerate']
            backend = None
        else:
            backend, systemsr = _get_system_sr()
            if sr is None:
                if not systemsr:
                    raise RuntimeError(f"sr for backend {backend} couldn't be determined")
                sr = systemsr

        logger.debug(f">>>>>>>>>>>>>>>>>>>> backend: {backend}, sr: {sr}")

        self.dt = dt = dt if dt is not None else _estimate_dt(sp)
        if start is None:
            start = sp.t0
        if end is None:
            end = sp.t1
        self.sr = sr
        self.isplaying = False
        self.playhead = -1
        self._interactive = interactive
        
        mtxfile = _spectrum_to_matrix(sp, dt=dt)
        atexit.register(_remove_if_exists, mtxfile)

        if interactive:
            self.oscport = _get_free_oscport()
            self.oscserver = liblo.ServerThread()
            notifyport = self.oscserver.port
            self._setup_oscserver()
            self.oscserver.start()
        else:
            self.oscport = None
            self.oscserver = None
            notifyport = None

        if verbose is None:
            verbose = logger.level <= logging.DEBUG

        self.process = _csound_player_process(
            mtxfile, 
            sr=sr, 
            backend=backend, 
            autoplay=autoplay, 
            looping=looping,
            exitwhendone=exitwhendone, 
            oscport=self.oscport, 
            speed=speed, 
            start=start, 
            end=end,
            oscnotify=notifyport,
            oscfreq=oscfreq,
            gain=gain,
            verbose=verbose)
        self._exited = False
        self._onpos = on_pos
        self._onplay = on_play

    def _setup_oscserver(self):
        def osc_pos(path, data):
            pos = data[0]
            self.playhead = pos
            callback = self._onpos
            if callback is not None:
                callback(pos)
        self.oscserver.add_method("/pos", "f", osc_pos)

        def osc_play(path, data):
            isplaying = data[0]
            self.isplaying = isplaying
            callback = self._onplay
            if callback is not None:
                callback(isplaying)
        self.oscserver.add_method("/play", "i", osc_play)

    def play(self, state):
        liblo.send(self.oscport, "/play", int(state))

    def stop(self):
        self.play(0)

    def set_looping(self, state):
        liblo.send(self.oscport, "/loop", int(state))

    def exit(self):
        if self._interactive:
            liblo.send(self.oscport, "/exit", 1)
            self.oscserver.free()
        if self.process is not None:
            time.sleep(0.1)
            self.process.terminate()
            self.process = None
        
    def set_position(self, t, dur=0):
        if dur == 0:
            liblo.send(self.oscport, "/setpos", t)
        else:
            liblo.send(self.oscport, "/setposline", t, dur)

    def set_playhead(self, t, dur):
        liblo.send(self.oscport, "/setplayhead", t, dur)

    def set_selection_end(self, t):
        liblo.send(self.oscport, "/setendpos", float(t))

    def set_speed(self, speed):
        liblo.send(self.oscport, "/speed", speed)

    def set_filter(self, active, freq0=0, freq1=24000):
        liblo.send(self.oscport, "/filter", int(active), float(freq0), float(freq1))

    def set_gain(self, gain=-1, noisegain=-1):
        liblo.send(self.oscport, "/gain", float(gain), float(noisegain))

    def set_bwfilter(self, bw0=-1, bw1=-1):
        logger.debug("set_bwfilter bw0={bw0}, bw1={bw1}")
        liblo.send(self.oscport, "/bwfilter", bw0, bw1)

    def is_active(self):
        if self._exited:
            return False
        running = self.process.poll() is None
        if not running:
            self._exited = True
        return running

    def __del__(self):
        logger.debug("__del__ SpectrumPlayer")
        self.exit()


def _csound_player_process(matrixpath, sr, backend=None, sndfile=None, 
                           autoplay=False, looping=False, oscport=None, speed=None, 
                           start=None, end=None, exitwhendone=False, oscnotify=None,
                           stereo=True, abstime=True, oscfreq=None, gain=None, quiet=True,
                           verbose=None):
    import pkg_resources as pkg
    matrixpath = os.path.abspath(matrixpath)
    csd = pkg.resource_filename(
        pkg.Requirement.parse("sndtrck"), "sndtrck/spectrum-player.csd")
    if verbose is None:
        verbose = logger.level <= logging.DEBUG
    render = sndfile is not None
    config = getconfig()
    if render:
        autoplay = True
        exitwhendone = True
        looping = False
        beadsyntflags = config['csound.render.beadsyntflags']
        stereo = False
        # -f -> float samples (32 bit)
        args = ['csound', '-o%s' % sndfile, '-r %d' % sr, '-f', 
                '--omacro:MATRIXPATH=%s' % matrixpath]
    else:
        beadsyntflags = config['csound.realtime.beadsyntflags']
        args = ["csound", "-odac", "-r %d" % sr,
                '--omacro:MATRIXPATH=%s' % matrixpath]
        if backend is None:
            backend = _get_audio_backend()
        args.append(f"-+rtaudio={backend}")
        if backend in _backends_which_need_realtime:
            args.append('--realtime')
        if quiet:
            args.extend(['-d', '-m', '0'])
    
    if autoplay:
        args.append("--omacro:AUTOPLAY=1")
    if looping:
        args.append("--omacro:LOOPMODE=%d" % int(looping))
    if oscport is not None:
        args.append("--omacro:OSCPORT=%d" % oscport)
    if speed is not None:
        args.append("--omacro:SPEED=%f" % speed)
    if start is not None:
        args.append("--omacro:POSITION=%f" % start)
    if end is not None: 
        args.append("--omacro:ENDPOS=%f" % end)
    if oscnotify is not None:
        args.append("--omacro:NOTIFYPORT=%d" % oscnotify)
    if exitwhendone:
        args.append("--omacro:EXITWHENDONE=%d" % int(exitwhendone))
    if stereo:
        args.append("--omacro:STEREO=1")
    if abstime:
        args.append("--omacro:ABSTIME=%d" % int(abstime))
    if gain is not None:
        args.append("--omacro:MASTERGAIN=%f" % gain)

    args.append("--omacro:BEADSYNTFLAGS=%d" % beadsyntflags)
    args.append(csd)
    
    logger.info("Csound command line: \n" + " ".join(args))
    if verbose:
        logger.debug("*********** Launching csound process without pipes")
        sub = subprocess.Popen(args)
    else:
        sub = subprocess.Popen(args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    pid = sub.pid
    atexit.register(lambda pid=pid:_kill_if_running(pid))
    return sub
    

def _kill_if_running(pid, killcode=9):
    """
    Kill pid if running. Returns True if it was running
    """
    try:
        os.kill(pid, killcode)
        return True
    except ProcessLookupError:
        return False


def _csound_render_spectrum(sp, sr, outfile, start=-1, end=-1):
    mtxpath = _spectrum_to_matrix(sp)
    subp = _csound_player_process(mtxpath, sr=sr, sndfile=outfile, start=start, end=end)
    try:
        subp.wait()
    finally:
        subp.terminate()


def _csound_render_spectrum_samples(sp, sr, start=-1, end=-1):
    sndfile = tempfile.mktemp(suffix=".wav", prefix="csound-render--")
    if os.path.exists(sndfile):
        logger.debug("Rendering spectrum to soundfile: {sndfile}, overwriting")
    _csound_render_spectrum(sp, sr=sr, outfile=sndfile, start=start, end=end)
    if not os.path.exists(sndfile):
        raise RuntimeError(f"Error rendering spectrum, sndfile {sndfile} not created")
    samples, sr2 = sndread(sndfile)
    assert int(sr) == int(sr2)
    return samples 


def render_sndfile(sp, sr, outfile, start=-1, end=-1, fade=None, 
                   method=None):
    """
    Render spectrum `sp` to a soundfile `outfile`

    See also: render_samples
    """
    if method is None:
        method = getconfig()['render.method']
    if method == 'loristrck' or method == 'loris':
        samples = render_samples(sp, sr, fade=fade, start=start, end=end, method='loris')
        return sndwrite(samples, sr, outfile)
    elif method == 'csound':
        return _csound_render_spectrum(sp, sr=sr, outfile=outfile, start=start, end=end)


def render_samples(sp, sr, start=-1, end=0, fade=None, method=None):
    """
    Render a spectrum, returns the samples as a numpy array

    sp: a Spectrum
    sr: the sr to render the spectrum
    fade: fade time (None to use a default)
    method: possible methods: loris, csound.
            If None, use configure method (key='render.method')
    
    See also: render_sndfile

    NB: the output array represent the rendered fragment: if asked
        to render with start=2, end=3 the array will hold 1 second
        of audio. To preserve the timing, a 2 second silence needs
        to be prepended
    """
    if start < 0:
        start = sp.t0
    if end <= 0:
        end = sp.t1
    if method is None:
        method = getconfig()['render.method']
    logger.debug(f"render_samples: start={start}, end={end}, method={method}")
    if method.startswith('loris'):
        from . import backend_loris
        if not backend_loris.is_available():
            raise RuntimeError("render: backend loris is not available")
        if start > sp.t0 or end < sp.t1:
            logger.debug(f"render_samples: cropping spectrum between {start} and {end}")
            sp = Spectrum([p.slice(start, end, include=True) for p in sp.partials_between(start, end)])
            # sp = sp.partials_between(start, end, crop=True)
        matrices = sp.asarrays()
        samples = backend_loris.synthesize(matrices, samplerate=sr, fadetime=fade, 
                                           start=start, end=end)
        return samples
    elif method == 'csound':    
        return _csound_render_spectrum_samples(sp, sr, start=start, end=end)