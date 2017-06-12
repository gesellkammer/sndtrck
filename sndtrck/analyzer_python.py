"""
Pure python implementation of Partial Tracking analysis.

NOT READY!
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from bpf4 import bpf
from numbers import Number
import numpy as np
from math import *
from scipy.special import i0, i1
PI = np.pi
DEBUG = True

class EnvelopeBuilder(object):
    def __init__(self):
        self._env = None
        self.xs = []
        self.ys = []
    @property
    def env(self):
        if self._env is not None:
            return self._env
        self._env = bpf.core.Linear(self.xs, self.ys)
        return self._env

class AmpEnvBuilder(EnvelopeBuilder):
    def build(self, peaks, frame_time):
        self.xs.append(frame_time)
        y = sqrt([peak.amplitude ** 2 for peak in peaks])
        self.ys.append(y)

def next_power_of_2(N):
    return int(ceil(log(N) / log(2.)))

def make_complex(real, imag):
    return real + 1j * imag

def time_ramped(vec):
    offset = 0.5 * (vec.size - 1)
    k = np.arange(0, vec.size)
    return vec * (k - offset)

class FourierTransform:
    def __init__(self, size):
        pass

class ReassignedSpectrum(object):
    def __init__(self, window, window_deriv):
        # self.window = window
        # self.window_deriv = window_deriv
        self.magnitude_transform = FourierTransform(1 << (1 + next_power_of_2( window.size )))
        self.correction_transform = FourierTransform(1 << (1 + next_power_of_2( window.size )))
        self.build_reassignment_windows(window, window_deriv)

    def build_reassignment_windows(self, window, window_deriv):
        winsum = window.sum()
        self.Window = window * (2/winsum)
        fancy_scale = window_deriv.size / (winsum * np.pi)
        framp = window_deriv * fancy_scale
        tramp = time_ramped(self.Window)
        tframp = time_ramped(framp)
        self.cplx_win_wd_wt = make_complex(framp, tramp)
        self.cplx_win_w_wtd = make_complex(self.Window, tframp)

    def apply_time_ramp(self, vec):
        offset = 0.5 * (vec.size - 1)
        k = np.arange(0, vec.size)
        vec *= (k - offset)

    @property
    def size(self):
        return self.magnitude_transform.size

    def transform(self, samples):
        pass
    def reassigned_frequency(self, idx):
        pass
    def reassigned_magnitude(self, idx):
        pass
    def convergence(self, idx):
        pass
    def reassigned_phase(self, idx):
        pass
    def reassigned_time(self, idx):
        pass
    def frequency_correction(self, sample):
        pass
    def time_correction(self, sample):
        pass

def zeroeth_order_bessel(x):
    return i0(x)

def first_order_bessel(x):
    return i1(x)

class KaiserWindow(object):
    @staticmethod
    def compute_shape(atten):
        if atten < 0:
            raise ValueError("Kaiser window shape must be positive")
        if atten > 60:
            alpha = 0.12438 * (atten + 6.3)
        elif atten > 13.26:
            alpha = 0.76609 * ( pow((atten - 13.26), 0.4) ) + 0.09834 * (atten - 13.26)
        else:
             alpha = 0.0
        return alpha

    @staticmethod
    def compute_length(width, alpha):
        """
        width: (0-1). Width of the window in Hz normalized to the samplerate
                    width = width_in_Hz / sr
        alpha: the result of compute_shape

        returns the size of the window in samples
        """
        return int(1.0 + (2. * sqrt((PI*PI) + (alpha*alpha)) / (PI * width)))

    @staticmethod
    def build_window(size, shape, method='numpy'):
        if method == 'numpy':
            return np.kaiser(size, shape)
        else:
            N = size - 1
            n = np.arange(0, size)
            K = (2 * n * (1/N)) - 1
            A = np.sqrt(1 - (K*K))
            win = zeroeth_order_bessel(shape*A) * (1/zeroeth_order_bessel(shape))
            return win

    @staticmethod
    def build(width, atten=90):
        alpha = KaiserWindow.compute_shape(atten)
        length = KaiserWindow.compute_length(width, alpha)
        return KaiserWindow.build_window(length, alpha)

    @staticmethod
    def build_time_derivative_window(winsize, shape):
        N = winsize - 1
        n = np.arange(0, winsize)
        K = 2*n*(1/N) - 1
        A = np.sqrt(1 - (K*K))
        commonFac = - 2*shape / (N*zeroeth_order_bessel(shape))
        win = commonFac * first_order_bessel(shape*A) * K/A
        win[0] = win[N] = 0
        return win

def kaiserwin(atten=90, width=80, sr=48000, must_be_odd=True):
    """
    atten: attenuantion in positive dB
    width: width of the window in Hz
           You can calculate the duration of the window as width/sr
    sr   : sample rate in Hz
    must_be_odd: make sure that the window has an odd number of samples 
                 (required by the Band-reassignment algorithm)
    """
    shape = KaiserWindow.compute_shape( self.sidelobe_level )
    size = KaiserWindow.compute_length(self.window_width / sr, winshape)
    if must_be_odd and not(size % 2):
        size += 1
    win = KaiserWindow.build_window(size, shape)
    return win

def kaiserwin_deriv(size, atten=90):
    """
    calculate the time derivative of the window

    size: in samples
    atten: in positive dB
    """
    shape = KaiserWindow.calculate_shape(atten)
    return KaiserWindow.build_time_derivative_window(size, shape)

def kaiser_win_and_deriv(atten=90, width=80, sr=48000, must_be_odd=True):
    """
    return the kaiser windows and its time derivative
    """
    win = kaiserwin(atten, width, sr, must_be_odd)
    deriv = kaiserwin_deriv(win.size, atten)
    return win, deriv

class Analyzer(object):
    def __init__(self, resolution, window_width=None):
        if window_width is None:
            if isinstance(resolution, Number):
                window_width = resolution * 2.0
        self.configure(resolution, window_width)
    def configure(self, resolution, window_width):
        """
        resolution (Hz):
        window_width (Hz):
        """
        if not callable(resolution):
            self.freq_resolution = resolution
            self.amp_floor = -90
            self.window_width = window_width
            self.sidelobe_level = -self.amp_floor
            self.freq_floor = resolution
            self.freq_drift = .5 * resolution
            self.hop_time = 1 / self.window_width
            self.crop_time = self.hop_time        
            self.store_residue_bandwidth()
            self.build_fundamental_env(0.99 * resolution, 1.5 * resolution)
            self.phase_correct = True    
        else:
            raise NotImplemented
            #resolution = bpf.asbpf(resolution)
        self.amp_env_builder = AmpEnvBuilder()

    def store_residue_bandwidth(self, region_width=2000):
        assert region_width > 0
        self.bw_assoc_param = region_width

    def build_fundamental_env(self, fmin, fmax, thresh_db= -60, thresh_hz=8000):
        pass

    def analyze(self, samples, sr, reference=None):
        if reference is None:
            reference = bpf.const(1)
        window, window_deriv = kaiser_win_and_deriv(
            self.sidelobe_level, self.window_width, sr, must_be_odd=True
        )
        spectrum = ReassignedSpectrum(window, window_deriv)

def debug(s):
    if DEBUG:
        print(s)

def analyze(samples, sr, **options):
    pass

