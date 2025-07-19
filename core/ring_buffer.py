from config.config_manager import ConfigManager
import numpy as np

class RingBuffer:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.window_size_samples = int(config.vad_config.window_size_sec * config.audio_config.sample_rate)
        self.stride_samples = int(config.vad_config.stride * config.audio_config.sample_rate)
        self.buffer_size = int(config.vad_config.buffer_size_sec * config.audio_config.sample_rate)
        self.buffer = np.zeros(self.buffer_size, dtype=config.audio_config.dtype)
        
        self.write_position = 0
        self.window_position = 0
        
    def addChunk(self, chunk: np.ndarray) -> None:
        chunk_len = len(chunk)

        if chunk_len > len(self.buffer):
            raise ValueError('Chunk size too big')
        
        end_position = self.write_position + chunk_len
        if end_position < self.buffer_size:
            self.buffer[self.write_position:end_position] = chunk
        else:
            first_part_len = self.buffer_size - self.write_position
            self.buffer[self.write_position:] = chunk[:first_part_len]
            self.buffer[:chunk_len - first_part_len] = chunk[first_part_len:]

        self.write_position = (self.write_position + chunk_len) % self.buffer_size

    def get_window(self):
        start = self.window_position
        end = (start + self.window_size_samples) % self.buffer_size

        if end > start:
            window = self.buffer[start:end]
        else:
            window = np.concatenate((self.buffer[start:], self.buffer[:end]))
        
        self.window_position = (self.window_position + self.stride_samples) % self.buffer_size

        return window
