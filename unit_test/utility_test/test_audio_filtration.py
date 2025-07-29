import pytest
import numpy as np
from scipy.signal import sosfilt
from utility.audio_filtration import (
    nyquist_freq_gen, highpass_filter, lowpass_filter, bandpass_filter,
    filter_audio, normalize_audio, CLIP
)

def test_nyquist_freq_gen():
    assert nyquist_freq_gen(16000) == 8000.0
    assert nyquist_freq_gen(44100) == 22050.0

def test_highpass_filter_output_shape():
    sos = highpass_filter(16000, 100)
    assert sos.shape[1] == 6  # SOS format
    assert isinstance(sos, np.ndarray)

def test_lowpass_filter_output_shape():
    sos = lowpass_filter(16000, 3000)
    assert sos.shape[1] == 6
    assert isinstance(sos, np.ndarray)

def test_bandpass_filter_output_shape():
    sos = bandpass_filter(16000, 300, 3000)
    assert sos.shape[1] == 6
    assert isinstance(sos, np.ndarray)

def test_filter_audio_behavior():
    fs = 16000
    t = np.linspace(0, 1, fs, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    sos = lowpass_filter(fs, 1000)
    filtered = filter_audio(audio, sos)

    assert filtered.shape == audio.shape
    assert filtered.dtype == np.float32
    assert np.max(np.abs(filtered)) <= CLIP

def test_filter_audio_clipping():
    audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
    sos = np.array([[1, 0, 0, 1, 0, 0]])  # identity filter (bypass)

    filtered = filter_audio(audio, sos)
    expected = np.clip(audio, -CLIP, CLIP)

    assert np.allclose(filtered, expected)

def test_normalize_audio_peak():
    audio = np.array([0.1, -0.5, 0.3], dtype=np.float32)
    normalized = normalize_audio(audio, target_peak=1.0)
    assert np.max(np.abs(normalized)) == pytest.approx(1.0)

def test_normalize_audio_zero_peak():
    audio = np.zeros(100, dtype=np.float32)
    normalized = normalize_audio(audio, target_peak=0.5)
    assert np.all(normalized == 0)
