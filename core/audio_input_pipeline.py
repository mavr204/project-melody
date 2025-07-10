# Library
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
from threading import Thread, Event
import queue

# Local
from core.VAD import SpeechVAD
import stubs.wake_up_detection as wad
from config.input_pipe_config import AudioConfig, VADConfig

def transcribe_audio(model: WhisperModel, audio: np.ndarray, beam_size: int) -> str:
    segments, info = model.transcribe(audio=audio, beam_size=beam_size)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    transcribed_text = ''
    for i, segment in enumerate(segments):
        print("[%.2fs -> %.2fs]: Segment %d" % (segment.start, segment.end, i + 1))
        transcribed_text += segment.text
    return transcribed_text

def transcribe_byte_audio(model: WhisperModel, beam_size: int, byte_audio: list | None = None) -> str:
    audio_float32 = np.frombuffer(b''.join(byte_audio), dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max
    return transcribe_audio(model=model, audio=audio_float32, beam_size=beam_size)

def record_audio_stream(config: AudioConfig, audio_queue: queue.Queue, stop_event: Event) -> None:
    sample_rate = config.sample_rate
    channels = config.channels
    dtype = config.dtype
    frame_duration_sec = config.duration
    frame_samples = int(sample_rate * frame_duration_sec)

    if channels != 1:
        raise ValueError("Only mono audio (1 channel) supported for VAD.")
    
    # Discard first frame to remove startup noise
    sd.rec(frame_samples, samplerate=sample_rate, channels=channels, dtype=dtype)
    sd.wait()

    print("Starting recording...")

    while not stop_event.is_set():
        recording = sd.rec(frame_samples, samplerate=sample_rate, channels=channels, dtype=dtype)
        sd.wait()
        recording = recording.flatten() 
        audio_queue.put(recording)
    
def monitor_voice_activity(vad_config: VADConfig, audio_queue: queue.Queue, stop_event: Event, model: WhisperModel, beam_size: int) -> bool:
    # Calculate the number of frames in the frame_duration
    frame_size = int(vad_config.sample_rate * (vad_config.frame_duration_ms / 1000.0) * np.dtype(np.int16).itemsize) # Frame duration in ms / 1000 = frame duration in seconds
    speech_buffer = []
    speech_detected = False
    silence_counter = 0
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
                silence_counter = 0
            elif speech_detected:
                print("silence counter: ", silence_counter)
                silence_counter += 1
                if silence_counter > vad_config.silence_counter_max:

                    stop_event.set()
                    print("VAD Stopped.")
                    print(transcribe_byte_audio(model=model, beam_size=beam_size, byte_audio=speech_buffer))
                    return  wad.wake_up_detection_stub(transcribe_byte_audio(model=model, beam_size=beam_size, byte_audio=speech_buffer))

def detect_voice(model: WhisperModel, vad_config: VADConfig, audio_config: AudioConfig, beam_size: int) -> bool:
    # Variables shared between two threads
    audio_queue = queue.Queue()
    stop_event = Event()
    
    recording_thread = Thread(target=record_audio_stream, args=(audio_config, audio_queue, stop_event))
    recording_thread.start()

    wake_word_detected = monitor_voice_activity(vad_config=vad_config, 
                                                audio_queue=audio_queue, 
                                                stop_event=stop_event, 
                                                model=model, 
                                                beam_size=beam_size)
    recording_thread.join()
    return wake_word_detected
