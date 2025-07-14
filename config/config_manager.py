from config.input_pipe_config import AudioConfig, VADConfig, WhisperModelConfig, VoiceBiometricConfig
from faster_whisper import WhisperModel

class ConfigManager:
    audio_config = None
    vad_config = None
    model_config = None

    def __init__(self):
        self.audio_config = AudioConfig()

        self.vad_config = VADConfig()
        
        self.model_config = WhisperModelConfig()
        self.model_config.model_sm = self.load_model()

        self.biometric_config = VoiceBiometricConfig()

    def load_model(self):
        model = WhisperModel(
            self.model_config.model_size,
            device=self.model_config.device,
            compute_type=self.model_config.compute_type
        )
        print("Model Loaded.")
        return model