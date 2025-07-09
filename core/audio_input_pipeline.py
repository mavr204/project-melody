# Library
import time
start = time.time()
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import datetime
import threading

# Local
from faster_whisper import WhisperModel
from config.input_pipe_config import AudioConfig, WhisperModelConfig, VADConfig
from core.VAD import SpeechVAD

print("Imports took ~", time.time() - start, "seconds")     

def record_audio(config:AudioConfig, write=False):
    duration = config.duration
    sample_rate = config.sample_rate
    channels = config.channels
    dtype = config.dtype
    def save_recording(sample_rate, recording):
        os.makedirs("./samples", exist_ok=True)
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        existing_files = [f for f in os.listdir("./samples") if f.endswith(".wav") and 'wavf' in f]
        serials = [int(f[4:7]) for f in existing_files if f[4:7].isdigit()]
        next_num = max(serials, default=0) + 1
        filename = f"./samples/wavf{next_num:03d}_{date_str}.wav"

        if recording.dtype == np.float32:
            wav.write(filename, sample_rate, (recording * np.iinfo(np.int16).max).astype(np.int16))
        else:
            wav.write(filename, sample_rate, recording)
        print(f"Saved as {filename}")

    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype=dtype)
    sd.wait()
    print("Recording finished.")
    recording[:int(0.5 * sample_rate)] = 0  # Remove noise from start
    
    if write:
        save_recording(sample_rate, recording)
    
    # Flattens the ndarray, for mono audio
    if channels == 1:
        recording = recording.flatten()
    print(recording.shape, recording.dtype, np.max(recording), np.min(recording))
    return recording

def load_model(config:WhisperModelConfig):
    model_size=config.model_size
    device=config.device
    compute_type=config.compute_type

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print("Model Loaded.")
    return model

def transcribe_audio(model, audio):
    segments, info = model.transcribe(audio=audio, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    transcribed_text = ''
    for i, segment in enumerate(segments):
        print("[%.2fs -> %.2fs]: Segment %d" % (segment.start, segment.end, i + 1))
        transcribed_text += segment.text
    return transcribed_text

def transcribe_live(model, audio_config)->str:
    audio = None

    audio = record_audio(audio_config)

    text = transcribe_audio(model, audio)
    return text

def voice_activity_detector(recording, model, vad_config):
    vad = SpeechVAD(vad_config)
    audio_bytes = (recording * np.iinfo(np.int16).max).astype(np.int16).tobytes()
    sample_rate = vad_config.sample_rate
    frame_duration_ms = vad_config.frame_duration_ms

    bytes_per_sample = np.dtype(np.int16).itemsize  # 2 bytes for int16
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * bytes_per_sample)

    speech_frames = []
    for i in range(0, len(audio_bytes), frame_size):
        frame = audio_bytes[i:i + frame_size]
        if len(frame) < frame_size:
            continue  # Skip incomplete frames
        if vad.isSpeech(frame):
            speech_frames.append(frame)
            print(transcribe_audio(model, recording))
            return False