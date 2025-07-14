# Library
import sounddevice as sd
import numpy as np
from threading import Thread, Event
import queue

# Local
from core.VAD import SpeechVAD
import stubs.wake_up_detection as wad
from config.config_manager import ConfigManager
from config.input_pipe_config import AudioConfig, VADConfig

def record_audio(duration: int) -> np.ndarray:
    config = AudioConfig(duration=duration)
    frames_in_record_duration = (config.duration * config.sample_rate)
    
    print('recording...')
    recording = sd.rec(frames=frames_in_record_duration, samplerate=config.sample_rate, channels=config.channels, dtype=config.dtype)
    sd.wait()
    print('recorded.')
    return recording

def transcribe_audio(config: ConfigManager, audio: np.ndarray) -> str:
    model_config = config.model_config
    segments, info = model_config.model_sm.transcribe(audio=audio, beam_size=model_config.beam_size)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    transcribed_text = ''
    for i, segment in enumerate(segments):
        print("[%.2fs -> %.2fs]: Segment %d" % (segment.start, segment.end, i + 1))
        transcribed_text += segment.text
    return transcribed_text

def byte_to_float32_audio(byte_audio: list | None = None) -> np.ndarray:
    return np.frombuffer(b''.join(byte_audio), dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max

def record_audio_stream(config: AudioConfig, audio_queue: queue.Queue, stop_event: Event) -> None:

    frame_duration_sec = config.duration
    frame_samples = int(config.sample_rate * frame_duration_sec)

    if config.channels != 1:
        raise ValueError("Only mono audio (1 channel) supported for VAD.")
    
    # Discard first frame to remove startup noise
    sd.rec(frame_samples, samplerate=config.sample_rate, channels=config.channels, dtype=config.dtype)
    sd.wait()

    print("Starting recording...")

    while not stop_event.is_set():
        recording = sd.rec(frame_samples, samplerate=config.sample_rate, channels=config.channels, dtype=config.dtype)
        sd.wait()
        recording = recording.flatten() 
        audio_queue.put(recording)
    
def voice_activity_detector(vad_config: VADConfig, audio_queue: queue.Queue, stop_event: Event) -> np.ndarray:
    # Calculate the number of frames in the frame_duration
    frame_size = int(vad_config.sample_rate * (vad_config.frame_duration_ms / 1000.0) * np.dtype(np.int16).itemsize) # Frame duration in ms / 1000 = frame duration in seconds
    speech_buffer = []
    speech_detected = False
    silence_frame_counter = 0
    vad = SpeechVAD(vad_config)
    
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
                silence_frame_counter = 0
            elif speech_detected:
                if silence_frame_counter == 0:
                    print("Silence Detected")
                silence_frame_counter += 1
                if silence_frame_counter > vad_config.silence_counter_max:

                    stop_event.set()
                    print("VAD Stopped.")
                    return  byte_to_float32_audio(speech_buffer)

def detect_voice(config: ConfigManager) -> np.ndarray:
    # Variables shared between two threads
    audio_queue = queue.Queue()
    stop_event = Event()
    
    recording_thread = Thread(target=record_audio_stream, args=(config.audio_config, audio_queue, stop_event))
    recording_thread.start()

    wake_word_detected = voice_activity_detector(vad_config=config.vad_config, 
                                                audio_queue=audio_queue, 
                                                stop_event=stop_event)
    recording_thread.join()
    return wake_word_detected
