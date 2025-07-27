from scipy.signal import butter, sosfilt
import numpy as np

"""
 Nyquist frequency is half the sampling rate of a signal.  
 It's the highest frequency that can be accurately represented without aliasing.

 CLIP is 1.0 because, that is the highest value in float32 audio data.
"""

CLIP=1.0


def nyquist_freq_gen(sample_rate: int) -> float:
    return sample_rate * 0.5

def highpass_filter(sample_rate: int, cutoff: int):
    normalized_freq = cutoff/nyquist_freq_gen(sample_rate=sample_rate)

    return butter(N=1, Wn=normalized_freq, btype='high', analog=False, output='sos')

def lowpass_filter(sample_rate: int, cutoff: int):
    normalized_freq = cutoff/nyquist_freq_gen(sample_rate=sample_rate)

    return butter(N=1, Wn=normalized_freq, btype='low', analog=False, output='sos')

def bandpass_filter(sample_rate: int, low_cutoff: int, high_cutoff: int):
    normalized_low = low_cutoff/nyquist_freq_gen(sample_rate=sample_rate)
    normalized_high = high_cutoff/nyquist_freq_gen(sample_rate=sample_rate)

    return butter(N=1, Wn=[normalized_low, normalized_high], btype='band', analog=False, output='sos')

def filter_audio(audio: np.ndarray, sos_filter) -> np.ndarray:
    data = sosfilt(sos_filter, audio)
    data = data.astype(np.float32)

    return np.clip(data, -CLIP, CLIP)

def normalize_audio(audio: np.ndarray, target_peak: float) -> np.ndarray:
    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio
    
    return (audio / peak) * target_peak
