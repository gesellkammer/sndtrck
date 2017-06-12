"""
SIMPL backend
"""

from __future__ import absolute_import


def analyze_simpl(samples, sr, resolution, windowsize, hopsize):
    """
    frame_size -> window_size
    resolution -> resolution
    """
    import simpl
    pd = simpl.LorisPeakDetection()
    pd.sampling_rate = sr
    pd.frame_size = windowsize
    pd.hop_size = hopsize
    maxpeaks = _simpl_resolution_to_maxpeaks(resolution, sr)
    pd.max_peaks = maxpeaks
    peak_frames = pd.find_peaks(samples)
    pt = simpl.LorisPartialTracking(maxpeaks)
    partial_frames = pt.find_partials(peak_frames)
    return partial_frames

def _simpl_maxpeaks_to_resolutioN(maxpeaks, sr):
    """
    convert the max. number of peaks to the resolution in Hz
    """
    return (sr / 2.) / maxpeaks

def _simpl_resolution_to_maxpeaks(resolution, sr):
    """
    convert the resolution of the analysis (in Hz) to the number of max. 
    number of peaks detectable
    """
    return (sr / 2.) / resolution

def analyze(samples, sr, resolution, windowsize, overlap=0.75):
    """
    resolution: analysis resolution in Hz
    windowsize: in samples
    overlap: 0-1. How much overlap between two analysis.
             an overlap of 0.5 determines a hopsize of windowsize*0.5
             an overlap of 0.75 determines a hopsize of windowsize*0.25
    """
    hopsize = windowsize * (1-overlap)
    return analyze_simpl(samples, sr, resolution, windowsize, hopsize)
