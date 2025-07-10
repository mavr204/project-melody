from config.input_pipe_config import AudioConfig, VADConfig, WhisperModelConfig
from faster_whisper import WhisperModel

class ConfigManager:
    def __init__(self):
        self.audio_config = AudioConfig()
        self.vad_config = VADConfig()
        self.model_config = WhisperModelConfig()
        self.whisper_model_sm = self.load_model()

    def load_model(self):
        model = WhisperModel(
            self.model_config.model_size,
            device=self.model_config.device,
            compute_type=self.model_config.compute_type
        )
        print("Model Loaded.")
        return model