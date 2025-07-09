import time
start = time.time()
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import datetime
import threading
from faster_whisper import WhisperModel
from ..config.input_pipe_config import AudioConfig, WhisperModelConfig

print("Imports took ~", time.time() - start, "seconds")     

# Configurations
audio_config = AudioConfig()
model_config = WhisperModelConfig()


def record_audio(audio_config:AudioConfig, write=False):
    duration = audio_config.duration
    sample_rate = audio_config.sample_rate
    channels = audio_config.channels
    dtype = audio_config.dtype
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

def load_model(model_config:WhisperModelConfig):
    model_size=model_config.model_size
    device=model_config.device
    compute_type=model_config.compute_type
    
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

def transcribe_live():
    audio = None
    model = None

    def thread_record():
        nonlocal audio
        audio = record_audio()

    def thread_model():
        nonlocal model
        model = load_model()

    t1 = threading.Thread(target=thread_record)
    t2 = threading.Thread(target=thread_model)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    text = transcribe_audio(model, audio)
    return text

def voice_activity_detector(aggression=10):
    pass