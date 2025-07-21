# Library
import sounddevice as sd
import numpy as np
from threading import Thread, Event
import queue

# Local
from core.VAD import SpeechVAD
import stubs.wake_up_detection as wad
from config.config_manager import ConfigManager
from config.input_pipe_config import AudioConfig

class InputPipeline:
    def __init__(self, config:ConfigManager):
        self.config = config
        self.queue = queue.Queue()
        self.stop_record_event = Event()

    def transcribe_audio(self, audio: np.ndarray) -> str:
        model_config = self.config.model_config
        segments, info = model_config.model_sm.transcribe(audio=audio, beam_size=model_config.beam_size)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        transcribed_text = ''
        for i, segment in enumerate(segments):
            print("[%.2fs -> %.2fs]: Segment %d" % (segment.start, segment.end, i + 1))
            transcribed_text += segment.text
        return transcribed_text
    
    def byte_to_float32_audio(self, byte_audio: list) -> np.ndarray:
        return np.frombuffer(b''.join(byte_audio), dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max
    
    def _record_audio_stream(self) -> None:
        config = self.config

        frame_duration_sec = config.audio_config.duration
        frame_samples = int(config.audio_config.sample_rate * frame_duration_sec)

        if config.audio_config.channels != 1: # Checks if the audio is mono 
            raise ValueError("Only mono audio (1 channel) supported for VAD.")
        
        # Discard first frame to remove startup noise
        sd.rec(frames=frame_samples,
               samplerate=config.audio_config.sample_rate, 
               channels=config.audio_config.channels, 
               dtype=config.audio_config.dtype)
        sd.wait()

        print("Starting recording...")

        while not self.stop_record_event.is_set():
            recording = sd.rec(frames=frame_samples,
                               samplerate=config.audio_config.sample_rate,
                               channels=config.audio_config.channels,
                               dtype=config.audio_config.dtype)
            sd.wait()
            recording = recording.flatten()


            self.audio_queue.put(recording)

    def _wake_up_validation(self, audio:np.ndarray):
            pass

    def _voice_activity_detector(self) -> np.ndarray:
        vad_config = self.config.vad_config
        frame_size = int(vad_config.sample_rate * (vad_config.frame_duration_ms / 1000.0) * np.dtype(np.int16).itemsize) # 1000ms = 1s
        speech_buffer = []
        speech_detected = False
        silence_frame_counter = 0
        vad = SpeechVAD(vad_config)
        wake_up_checks = {'wake_up_check':False, 'biometric_check':False}
        check_wake_up = True
        wake_up_audio = np.array([], dtype=np.float32)
        wake_up_check_thread = None
        
        print("VAD running...")
        while True:
            recording = self.queue.get()

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
                        break

            if check_wake_up and not speech_detected:
                if wake_up_audio.size > 2: # send 2 chunks of audio totaling 1 second of audio
                    wake_up_audio = np.concatenate((wake_up_audio, recording))
                else:
                    check_wake_up = False

                    wake_up_check_thread = Thread(target=self._wake_up_validation, args=(wake_up_audio, wake_up_checks))
                    wake_up_check_thread.start()

            if silence_frame_counter > vad_config.silence_counter_max:
                self.stop_record_event.set()
                print("VAD Stopped.")
                return  self.byte_to_float32_audio(speech_buffer)

    def detect_voice(self) -> np.ndarray:
        recording_thread = Thread(target=self._record_audio_stream)
        recording_thread.start()

        wake_word_detected = self._voice_activity_detector()
        recording_thread.join()
        return wake_word_detected