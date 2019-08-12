from spectrum import Spectrum
from emlib.snd import features
from .typehints import List

def onsets(s: Spectrum, threshold=0.03, mingap=0.050) -> List[float]:
    """
    Detect global onsets based on the spectrum

    Returns a list of seconds where each represents an onset
    """
    sr = 22050
    samples = s.synthesize(sr)
    onsets = features.onsets_aubio(samples, sr=sr, threshold=threshold, mingap=mingap)
    return onsets

