# Library
import sys
import sounddevice as sd
import numpy as np
from threading import Thread, Event
import queue

# Local
from core.VAD import SpeechVAD
from core.template_generator import BiometricTemplateGenerator
import stubs.wake_up_detection as wud
from config.config_manager import ConfigManager
from utility.logger import get_logger

class WakeUpChecks:
    def __init__(self):
        self.wake_up: bool = False
        self.biometric_pass: bool = False

logger = get_logger(__name__)

class InputPipeline:
    def __init__(self, config:ConfigManager, voice_template: BiometricTemplateGenerator):
        self._config = config
        self.queue = queue.Queue()
        self.stop_record_event = Event()
        self.voice_template = voice_template
        self.vad = SpeechVAD(self._config.vad_config)
        self.silence_frame_counter = 0

    def transcribe_audio(self, audio: np.ndarray) -> str:
        model_config = self._config.model_config
        segments, info = model_config.model_sm.transcribe(audio=audio, beam_size=model_config.beam_size)
        logger.info(f"Detected language '{info.language}' with probability {info.language_probability}")

        transcribed_text = ''
        for i, segment in enumerate(segments):
            logger.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s]: Segment {i+1}")
            transcribed_text += segment.text
        return transcribed_text
    
    def byte_to_float32_audio(self, byte_audio: list[bytes]) -> np.ndarray:
        return np.frombuffer(b''.join(byte_audio), dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max
    
    def _record_audio_stream(self) -> None:
        config = self._config

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

        logger.debug("Starting recording...")
        while not self.stop_record_event.is_set():
            recording = sd.rec(frames=frame_samples,
                               samplerate=config.audio_config.sample_rate,
                               channels=config.audio_config.channels,
                               dtype=config.audio_config.dtype)
            sd.wait()
            recording = recording.flatten()


            self.queue.put(recording)

    def wake_up_validation(self, audio:np.ndarray, wake_up_checks: WakeUpChecks) -> None:
            wake_up_checks.wake_up = wud.wake_up_detection_stub(ip=self.transcribe_audio(audio=audio))

            if wake_up_checks.wake_up:
                wake_up_checks.biometric_pass = self.voice_template.match_embedding()

    def _process_audio_to_bytes(self) -> bytes:
        try:
            chunk = self.queue.get(timeout=5)
            return (chunk * np.iinfo(np.int16).max).astype(np.int16).tobytes()
        except queue.Empty:
            logger.critical("Queue is empty, Exiting...")
            self.stop_record_event.set()
            sys.exit(1)

    def _voice_activity_detector(self, speech_frames: list[bytes]) -> bool:
        audio_bytes = self._process_audio_to_bytes()
        frame_size = int(self._config.audio_config.sample_rate * (self._config.vad_config.frame_duration_ms / 1000.0) * np.dtype(np.int16).itemsize) # 1000ms = 1s
        speech_detected = False

        for i in range(0, len(audio_bytes), frame_size):
            frame = audio_bytes[i:i + frame_size]
            if len(frame) < frame_size:
                logger.error("Invalid Audio")
                continue
            if self.vad.isSpeech(frame):
                speech_frames.append(frame)
                speech_detected = True
                self.silence_frame_counter = 0
            elif speech_detected:
                if self.silence_frame_counter == 0:
                    logger.debug("Silence Detected")
                self.silence_frame_counter += 1
                
                if self.silence_frame_counter > self._config.vad_config.silence_counter_max:
                    speech_detected = False
                    break
        
        silence_detected = self.silence_frame_counter >= self._config.vad_config.silence_counter_max
        return speech_detected or (not silence_detected) # returns false when prolonged silence is detected from collected speech frames.

    def _wake_up_detect(self) -> np.ndarray: 
        speech_frames = []
        check_wake = True
        wake_up_check_thread = None
        wake_up_checks = WakeUpChecks()
        
        logger.debug("VAD running...")
        while True:
            vad_active = self._voice_activity_detector(speech_frames=speech_frames)

            if len(speech_frames) >= 33 and check_wake:
                audio = self.byte_to_float32_audio(speech_frames)
                wake_up_check_thread = Thread(target=self.wake_up_validation, args=(audio, wake_up_checks))
                wake_up_check_thread.start()
                check_wake = False
            
            if vad_active == False:
                if wake_up_check_thread:
                    wake_up_check_thread.join()

                if wake_up_checks.wake_up == False or wake_up_checks.biometric_pass == False:
                    if wake_up_checks.biometric_pass == False:
                        logger.warning('Biometric Failed!')
                    else:        
                        logger.warning('No wake up detected!')
                    
                    # Reset the --make an appropriate comment
                    speech_frames.clear()
                    check_wake = True
                    wake_up_checks = WakeUpChecks()
                    continue
                break

        return self.byte_to_float32_audio(speech_frames)

    def get_command(self) -> np.ndarray:
        recording_thread = Thread(target=self._record_audio_stream)
        recording_thread.start()

        wake_word_detected = self._wake_up_detect()
        recording_thread.join()
        return wake_word_detected