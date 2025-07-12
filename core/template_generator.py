from core.audio_input_pipeline import record_audio
import torch

class TemplateGen:
    def __init__(self, audio_config):
        pass

    def numpy_to_tensor(self, numpy_data, sample_rate):
        if sample_rate != 16_000:
            raise ValueError("ECAPA requires 16kHz audio.")
        return torch.tensor(numpy_data, dtype=torch.float32).unsqueeze(0)
    
    def get_embedding(self, audio_np):
        self.numpy_to_tensor(audio_np=)