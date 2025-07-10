from dataclasses import dataclass
from math import ceil

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
    beam_size = 5


@dataclass
class VADConfig:
    aggressiveness:int = 3
    sample_rate: int = 16_000
    frame_duration_ms = 30
    silence_time = 1
    silence_counter_max = int(ceil(1/(frame_duration_ms/1000))) # frame_duration_ms/1000 = frame_duration in seconds