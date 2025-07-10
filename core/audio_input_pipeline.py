# Library
import time
start = time.time()
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import datetime
import threading
import queue

# Local
from faster_whisper import WhisperModel
from config.input_pipe_config import AudioConfig, WhisperModelConfig, VADConfig
from core.VAD import SpeechVAD
import stubs.wake_up_detection as wad

print("Imports took ~", time.time() - start, "seconds")

def record_audio(config:AudioConfig, write=False)->np.ndarray:
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
    
    # Flattens the ndarray to 1 dimension, for mono audio
    if channels == 1:
        recording = recording.flatten()
    return recording

def load_model(config:WhisperModelConfig)->WhisperModel:
    model_size=config.model_size
    device=config.device
    compute_type=config.compute_type

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print("Model Loaded.")
    return model

def transcribe_audio(model, audio)->str:
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

def record_audio_stream(config: AudioConfig, audio_queue: queue.Queue, stop_event:threading.Event)->None:
    sample_rate = config.sample_rate
    channels = config.channels
    dtype = config.dtype
    frame_duration_sec = config.duration
    frame_samples = int(sample_rate * frame_duration_sec)

    if channels != 1:
        raise ValueError("Only mono audio (1 channel) supported for VAD.")
    
    print("Starting recording...")

    while not stop_event.is_set():
        recording = sd.rec(frame_samples, samplerate=sample_rate, channels=channels, dtype=dtype)
        sd.wait()
        recording = recording.flatten() 
        audio_queue.put(recording)

def voice_activity_detector(model, vad_config:VADConfig, audio_config:AudioConfig)->bool:
    vad = SpeechVAD(vad_config)
    sample_rate = vad_config.sample_rate
    frame_duration_ms = vad_config.frame_duration_ms
    bytes_per_sample = np.dtype(np.int16).itemsize  # 2 bytes for int16
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * bytes_per_sample)
    max_silence_frames = vad_config.silence_counter
    audio_queue = queue.Queue()

    stop_event = threading.Event()
    recording_thread = threading.Thread(target=record_audio_stream, args=(audio_config, audio_queue, stop_event))
    recording_thread.start()
    
    speech_buffer = []
    speech_detected = False
    silence_counter = 0
    
    print("VAD running...")
    while True:
        recording = audio_queue.get()

        audio_bytes = (recording * np.iinfo(np.int16).max).astype(np.int16).tobytes()

        for i in range(0, len(audio_bytes), frame_size):
            frame = audio_bytes[i:i + frame_size]
            if len(frame) < frame_size:
                continue 
            if vad.isSpeech(frame):
                speech_buffer.append(frame)
                speech_detected = True
                silence_counter = 0
            elif speech_detected:
                silence_counter += 1
                if silence_counter > max_silence_frames:

                    stop_event.set()
                    recording_thread.join()
                    speech_bytes = b''.join(speech_buffer)
                    recording_float32 = np.frombuffer(speech_bytes, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max                    
                    return  wad.wake_up_detection_stub(transcribe_audio(model, recording_float32))
