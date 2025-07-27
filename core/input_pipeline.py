# Library
import sys
import sounddevice as sd
import numpy as np
from threading import Thread, Event, enumerate as thread_enumerate
import queue

# Local
from core.VAD import SpeechVAD
from core.template_generator import BiometricTemplateGenerator
import stubs.wake_up_detection as wud
from config.config_manager import ConfigManager
from utility.logger import get_logger
import utility.errors as err
from utility.thread_manager import ThreadManager, ThreadStatus
from utility.audio_filtration import normalize_audio, bandpass_filter, filter_audio

class WakeUpChecks:
    def __init__(self):
        self.wake_up: bool = False
        self.biometric_pass: bool = False

logger = get_logger(__name__)

class InputPipeline:
    _STREAM_THREAD_NAME = 'AudioStreamThread'

    def __init__(self, config:ConfigManager, voice_template: BiometricTemplateGenerator):
        self._config = config
        self.queue = queue.Queue()
        self.thread_manager = ThreadManager()
        self.voice_template = voice_template
        self.vad = SpeechVAD(self._config.vad_config)
        self.silence_frame_counter = 0
        self.audio_filter = bandpass_filter(sample_rate=config.audio_config.sample_rate,
                                            low_cutoff=config.filter_config.low_cutoff,
                                            high_cutoff=config.filter_config.high_cutoff)

    def transcribe_audio(self, audio: np.ndarray) -> str:
        try:
            model_config = self._config.model_config
            segments, info = model_config.model_sm.transcribe(audio=audio, beam_size=model_config.beam_size)
            logger.info(f"Detected language '{info.language}' with probability {info.language_probability}")

            transcribed_text = ''
            for i, segment in enumerate(segments):
                logger.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s]: Segment {i+1}")
                transcribed_text += segment.text
            return transcribed_text
        except Exception as e:
            raise err.TranscriptionError("Failed to transcribe Audio") from e
    
    def byte_to_float32_audio(self, byte_audio: list[bytes]) -> np.ndarray:
        if not byte_audio:
            raise err.InvalidAudioError("Byte audio is not valid")
        return np.frombuffer(b''.join(byte_audio), dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max
    
    def _record_audio_stream(self) -> None:
        config = self._config

        frame_duration_sec = config.audio_config.duration
        frame_samples = int(config.audio_config.sample_rate * frame_duration_sec)

        if config.audio_config.channels != 1: # Checks if the audio is mono 
            raise ValueError("Only mono audio (1 channel) supported for VAD.")
        
        try:
            # Discard first frame to remove startup noise
            sd.rec(frames=frame_samples,
                samplerate=config.audio_config.sample_rate, 
                channels=config.audio_config.channels, 
                dtype=config.audio_config.dtype)
            sd.wait()

            logger.debug("Starting recording...")
            stop_event = self.thread_manager.active_threads.get(self._STREAM_THREAD_NAME).stop_event
            if stop_event is None:
                logger.critical(f"The stream must be run on {self._STREAM_THREAD_NAME}")
                raise err.AudioStreamError("Failed to stream audio")
            while not stop_event.is_set():
                recording = sd.rec(frames=frame_samples,
                                samplerate=config.audio_config.sample_rate,
                                channels=config.audio_config.channels,
                                dtype=config.audio_config.dtype)
                sd.wait()
                recording = recording.flatten()
                self.queue.put(filter_audio(audio=recording, sos_filter=self.audio_filter))
        except Exception as e:
            raise err.AudioStreamError("There was an Error recording audio...") from e

    def wake_up_validation(self, audio:np.ndarray, wake_up_checks: WakeUpChecks) -> None:
            transcript = self.transcribe_audio(audio=audio)
            audio = normalize_audio(audio=audio, target_peak=self._config.filter_config.normalizing_peak)
            logger.debug("Wake Up prompt: " + transcript)
            wake_up_checks.wake_up = wud.wake_up_detection_stub(ip=transcript)

            if wake_up_checks.wake_up:
                wake_up_checks.biometric_pass = self.voice_template.match_embedding(audio=audio)

    def _process_audio_to_bytes(self) -> bytes:
        try:
            chunk = self.queue.get(timeout=5)
            return (chunk * np.iinfo(np.int16).max).astype(np.int16).tobytes()
        except queue.Empty:
            logger.critical("Queue is empty, Exiting...")
            self._kill_all_child_threads()
            sys.exit(1)

    def _voice_activity_detector(self, speech_frames: list[bytes]) -> bool:
        audio_bytes = self._process_audio_to_bytes()
        frame_size = int(self._config.audio_config.sample_rate * (self._config.vad_config.frame_duration_ms / 1000.0) * np.dtype(np.int16).itemsize) # 1000ms = 1s
        speech_detected = False
        silence_detected = False

        for i in range(0, len(audio_bytes), frame_size):
            frame = audio_bytes[i:i + frame_size]

            if len(frame) < frame_size:
                logger.error("Invalid Audio")
                continue
            if self.vad.isSpeech(frame):
                speech_frames.append(frame)
                speech_detected = True
                self.silence_frame_counter = 0
            
            elif speech_detected or self.silence_frame_counter > 0:
                if self.silence_frame_counter == 0:
                    logger.debug("Silence Detected")
                self.silence_frame_counter += 1
                
                if self.silence_frame_counter >= self._config.vad_config.silence_counter_max:
                    logger.debug("stopping VAD")
                    self.silence_frame_counter = 0
                    silence_detected = True
                    speech_detected = False
                    break
        return speech_detected or (not silence_detected) # returns false when prolonged silence is detected from collected speech frames.

    def _wake_up_detect(self) -> np.ndarray: 
        speech_frames = []
        check_wake = True
        wake_up_check_thread = None
        wake_up_checks = WakeUpChecks()
        
        logger.debug("VAD running...")
        while True:
            vad_active = self._voice_activity_detector(speech_frames=speech_frames)

            if len(speech_frames) >= self._config.vad_config.check_wake_after_frames and check_wake:
                audio = self.byte_to_float32_audio(speech_frames)

                wake_up_check_thread = self.thread_manager.create_new_thread(target=self.wake_up_validation,
                                                                            args=(audio, wake_up_checks),
                                                                            name="WakeCheckThread",
                                                                            autostart=True)
                if wake_up_check_thread is None:
                    raise err.WakeUpError("Could not check for Error!")
                        
                check_wake = False
            
            if vad_active == False:
                if self.thread_manager.get_thread_status(wake_up_check_thread) != ThreadStatus.NOT_FOUND:
                    self.thread_manager.stop_thread(wake_up_check_thread)

                if wake_up_checks.wake_up == False or wake_up_checks.biometric_pass == False:
                    if not wake_up_checks.wake_up:
                        logger.warning('No wake up detected!')
                    else:        
                        logger.warning('Biometric Failed!')
                    
                    speech_frames.clear()
                    check_wake = True
                    wake_up_checks = WakeUpChecks()
                    continue
                break

        return self.byte_to_float32_audio(speech_frames)

    def get_command(self) -> np.ndarray:
        if not self.voice_template.is_template:
            raise err.TemplateLoadError("No biometric template found or loaded!")


        recording_thread = self.thread_manager.create_new_thread(target=self._record_audio_stream,
                                                                name=self._STREAM_THREAD_NAME,
                                                                autostart=True)
        
        if recording_thread is None:
            raise err.AudioStreamError("Audio could not be streamed")

        audio = self._wake_up_detect()

        try:
            self.thread_manager.stop_thread(recording_thread)
        except err.ThreadNotFoundError as e:
            logger.critical(f"Audio Not recorded: {e}")
            return np.array([], dtype=np.float32)

        self.queue = queue.Queue() # Reset the Queue
        return audio
    
    def get_template_audio(self) -> list[np.ndarray]:
        speech_frames: list[bytes] = []
        audio_samples_count = 0
        audio_samples: list[np.ndarray] = []
        self.thread_manager.stop_all_threads()

        logger.info("Recording Info for Template generation...")
        recording_thread = self.thread_manager.create_new_thread(target=self._record_audio_stream,
                                                                name=self._STREAM_THREAD_NAME,
                                                                autostart=True)
        
        if recording_thread is None:
            raise err.AudioStreamError("Audio could not be streamed")

        logger.debug("VAD running...")
        while audio_samples_count < self._config.biometric_config.audio_sample_required:
            vad_active = self._voice_activity_detector(speech_frames=speech_frames)

            if not vad_active:
                audio = self.byte_to_float32_audio(speech_frames)
                transcription = self.transcribe_audio(audio=audio)
                if wud.wake_up_detection_stub(ip=transcription):
                    audio = normalize_audio(audio=audio, target_peak=self._config.filter_config.normalizing_peak)
                    audio_samples.append(audio)
                    logger.info(f'detected prompt: {transcription}')
                    audio_samples_count += 1
                    logger.info(f'Audio Recorded: {audio_samples_count}/{self._config.biometric_config.audio_sample_required}')
                else:
                    logger.warning("That was not a valid keyword. Try again...")
                
                speech_frames.clear()
        
        try:
            self.thread_manager.stop_thread(recording_thread)
        except err.ThreadNotFoundError as e:
            logger.critical(f"Audio Not recorded: {e}")
            raise err.BiometricError("Failed to get audio")
        
        self.queue = queue.Queue() # Reset the Queue
        return audio_samples
    
    
