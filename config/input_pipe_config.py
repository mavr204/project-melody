from dataclasses import dataclass
from math import ceil
from faster_whisper import WhisperModel

@dataclass
class AudioConfig:
    duration: int = 0.5
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
    silence_time = 1
    silence_counter_max = int(ceil(1/(frame_duration_ms/1000))) # frame_duration_ms/1000 = frame_duration in seconds

@dataclass
class VoiceBiometricConfig:
    template_path: str = './template/voice_template.npy'
    audio_sample_required: int = 1
    threshold: float = 0.75