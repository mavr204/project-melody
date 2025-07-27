from scipy.signal import butter, sosfilt
import numpy as np

N=3
CLIP=1.0

def highpass_filter(sample_rate: int, cutoff: int):
    nyquist_freq = sample_rate / 2.0
    normalized_freq = cutoff/nyquist_freq

    return butter(N=N, Wn=normalized_freq, btype='high', analog=False, output='sos')

def lowpass_filter(sample_rate: int, cutoff: int):
    nyquist_freq = sample_rate / 2.0
    normalized_freq = cutoff/nyquist_freq

    return butter(N=N, Wn=normalized_freq, btype='low', analog=False, output='sos')

def bandpass_filter(sample_rate: int, low_cutoff: int, high_cutoff: int):
    nyquist_freq = sample_rate / 2.0
    normalized_low = low_cutoff/nyquist_freq
    normalized_high = high_cutoff/nyquist_freq

    return butter(N=N, Wn=[normalized_low, normalized_high], btype='band', analog=False, output='sos')

def filter_audio(audio: np.ndarray, sos_filter) -> np.ndarray:
    data = sosfilt(sos_filter, audio)
    data = data.astype(np.float32)
    # data = trim_low_energy(audio=data, sample_rate=16_000, threshold=0.0009, frame_duration_ms=30)

    # return np.clip(normalize_audio(data, target_peak=0.80), -CLIP, CLIP)
    return np.clip(data, -CLIP, CLIP)

def normalize_audio(audio: np.ndarray, target_peak: float) -> np.ndarray:
    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio
    
    return (audio / peak) * target_peak

def trim_low_energy(audio: np.ndarray, sample_rate: int, threshold: float = 0.01, frame_duration_ms: int = 20) -> np.ndarray:
    frame_length = int(sample_rate * frame_duration_ms / 1000)
    total_frames = len(audio) // frame_length

    mask = []
    for i in range(total_frames):
        frame = audio[i * frame_length : (i + 1) * frame_length]
        energy = np.mean(np.abs(frame))
        if energy > threshold:
            mask.append(frame)

    if mask:
        trimmed = np.concatenate(mask)
        return trimmed
    else:
        return np.array([], dtype=audio.dtype)  

