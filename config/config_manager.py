from config.input_pipe_config import AudioConfig, VADConfig, WhisperModelConfig, VoiceBiometricConfig
from faster_whisper import WhisperModel

class ConfigManager:
    audio_config = None
    vad_config = None
    model_config = None

    def __init__(self):
        self.vad_config = VADConfig()
        
        audio_chunk_duration = (self.vad_config.frame_duration_ms / 1000) * 16 # Rcord chunks 16 times the size of the frames used in vad, 1000ms = 1s
        self.audio_config = AudioConfig(duration=audio_chunk_duration)
        
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