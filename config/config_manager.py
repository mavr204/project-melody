import config.input_pipe_config as config
from faster_whisper import WhisperModel
import utility.errors as err

class ConfigManager:

    def __init__(self):
        self.basic_info = config.BasicInfo()

        self.vad_config = config.VADConfig()
        
        audio_chunk_duration = (self.vad_config.frame_duration_ms / 1000) * 16 # Rcord chunks 16 times the size of the frames used in vad, 1000ms = 1s
        self.audio_config = config.AudioConfig(duration=audio_chunk_duration)
        
        self.model_config = config.WhisperModelConfig()
        self.model_config.model_sm = self.load_model()

        self.biometric_config = config.VoiceBiometricConfig()

    def load_model(self):
        try:
            model = WhisperModel(
                self.model_config.model_size,
                device=self.model_config.device,
                compute_type=self.model_config.compute_type
            )
            print("Model Loaded.")
            return model
        except Exception as e:
            raise err.ModelLoadError("Failed to load model!") from e