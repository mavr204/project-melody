from dataclasses import dataclass
from math import ceil
import os
import sys
from faster_whisper import WhisperModel
from appdirs import user_data_dir
from utility.logger import get_logger

logger = get_logger(__name__)

class BasicInfo:
    def __init__(self):
        self._app_name = 'Melody'
        self._author_name = 'mav204'
        try:
            self._usr_data_dir = user_data_dir(appname=self._app_name, appauthor=self._author_name)
            os.makedirs(self._usr_data_dir, exist_ok=True)
        except PermissionError:
            logger.critical("Permission denied when creating user data directory.")
            sys.exit(1)
        except OSError as e:
            logger.critical(f"Failed to create user data directory: {e}")
            sys.exit(1)
    
    @property
    def app_name(self):
        return self._app_name
    
    @property
    def author_name(self):
        return self._author_name
    
    @property
    def usr_data_dir(self):
        return self._usr_data_dir

@dataclass
class AudioConfig:
    duration: int
    sample_rate: int = 16_000
    channels: int = 1
    dtype: str = 'float32'


@dataclass
class WhisperModelConfig:
    model_size:str = 'small'
    device:str = 'cpu'
    compute_type:str = 'int8'
    beam_size: int = 5
    model_sm :WhisperModel | None = None


@dataclass
class VADConfig:
    aggressiveness:int = 3
    sample_rate: int = 16_000
    frame_duration_ms = 30
    silence_time = 0.5
    silence_counter_max = int(ceil(1/(frame_duration_ms/1000))) # calculates the number of frames in 1 second, 1000ms = 1s
    
    window_size_sec: float = 1.5
    stride : float = 1
    buffer_size_sec: float = 5

@dataclass
class VoiceBiometricConfig:
    audio_sample_required: int = 3
    threshold: float = 0.75
    encrypt_key_length = 256
    nonce_bytes = 12 # generates a 12 nonce | AES-GCM standadizes a 12 byte nonce

    def get_file_name(self, username: str) -> str:
        return f"biometric_template_{username}.enc"
    
    def get_file_pattern(self) -> tuple:
        dummy_username = 'usr'
        file = self.get_file_name(dummy_username)

        return tuple(file.split(dummy_username))
    
    def extract_username(self, filename: str) -> str:
        prefix, suffix = self.get_file_pattern()
        if filename.startswith(prefix) and filename.endswith(suffix):
            return filename[len(prefix):-len(suffix)]
        else:
            raise ValueError("Invalid template filename format.")
