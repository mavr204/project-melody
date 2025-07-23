import config.config_manager
import numpy as np

config = config.config_manager.ConfigManager()

print(int(config.audio_config.sample_rate * (config.vad_config.frame_duration_ms / 1000.0) * np.dtype(np.int16).itemsize))