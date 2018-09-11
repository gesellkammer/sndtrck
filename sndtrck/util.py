import os
import numpy as np
import pysndfile


def aslist(x):
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
    return hasattr(obj, '__iter__') and not isinstance(obj, str)


def normalizepath(path):
    return os.path.abspath(os.path.expanduser(path))


def sndreadmono(sndfile, channel=0, start=0, end=0):
    """
    Read a soundfile. If the file has more than one channel
    returns the indicated channel. 

    If start and/or end are given, only a portion of the soundfile is read

    start: start time, in seconds
    end: end time in seconds
         0: read until the end
         negative numbers: time counted from the end 
                           (-2: read until 2 seconds before the end)
    """
    samples, sr = sndread(sndfile, start=start, end=end)
    mono = samples if len(samples.shape) == 1 else samples[:,channel]
    return mono, sr


def _convert_mp3_wav(mp3, wav, start=0, end=0):
    import subprocess
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Can't read mp3: ffmpeg is not present")
    cmd = [ffmpeg]
    if start > 0:
        cmd.extend(["-ss", str(start)])
    if end > 0:
        cmd.extend(["-t", str(end - start)])
    cmd.extend(["-i", mp3])
    cmd.append(wav)
    subprocess.call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _sndread_mp3(mp3, start=0, end=0):
    import tempfile
    wav = tempfile.mktemp(suffix=".wav")
    _convert_mp3_wav(mp3, wav, start=start, end=end)
    out = sndread(wav)
    os.remove(wav)
    return out


def sndread(sndfile, start=0, end=0):
    ext = os.path.splitext(sndfile)[1]
    if ext == '.mp3':
        return _sndread_mp3(sndfile, start=start, end=end)
    sf = pysndfile.PySndfile(sndfile)
    sr = sf.samplerate()
    duration = sf.frames() / sr
    if end <= 0:
        end = duration - end
    if start >= end:
        raise ValueError(f"Asked to read 0 frames: start={start}, end={end}")
    if start > 0:
        if start > duration:
            raise ValueError(f"Asked to read after end of file (start={start}, duration={duration}")
        sf.seek(int(start * sr))
    frames = sf.read_frames(int((end - start)*sr))
    return frames, sr


def sndwrite(samples, sr, sndfile, encoding=None):
    """
    encoding: 'pcm8', 'pcm16', 'pcm24', 'pcm32', 'flt32'. 
              None to use a default based on the given extension
    """
    ext = os.path.splitext(sndfile)[1].lower()
    if encoding is None:
        encoding = _defaultEncodingForExtension(ext)
    fmt = _getFormat(ext, encoding)
    snd = pysndfile.PySndfile(sndfile, mode='w', format=fmt,
                              channels=_numchannels(samples), samplerate=sr)
    snd.write_frames(samples)
    snd.writeSync()


def _getFormat(extension:str, encoding:str):
    assert extension[0] == "."
    fmt, bits = encoding[:3], int(encoding[3:])
    assert fmt in ('pcm', 'flt') and bits in (8, 16, 24, 32)
    extension = extension[1:]
    if extension == 'aif':
        extension = 'aiff'
    fmt = "%s%d" % (
        {'pcm': 'pcm', 
         'flt': 'float'}[fmt],
        bits
    )
    return pysndfile.construct_format(extension, fmt)


def _numchannels(samples:np.ndarray) -> int:
    """
    return the number of channels present in samples

    samples: a numpy array as returned by sndread

    """
    return 1 if len(samples.shape) == 1 else samples.shape[1]


def _defaultEncodingForExtension(ext):
    if ext == ".wav" or ext == ".aif" or ext == ".aiff":
        return "flt32"
    elif ext == ".flac":
        return "pcm24"
    else:
        raise KeyError(f"extension {ext} not known")


def freopen(f, option, stream):
    """
    freopen("hello", "w", sys.stdout)
    """
    oldf = open(f, option)
    oldfd = oldf.fileno()
    newfd = stream.fileno()
    os.close(newfd)
    os.dup2(oldfd, newfd)

