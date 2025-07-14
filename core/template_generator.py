import numpy as np
from resemblyzer import VoiceEncoder
from core.audio_input_pipeline import transcribe_audio, detect_voice
from config.config_manager import ConfigManager
import stubs.wake_up_detection as wad
from torch import tensor, float32, Tensor

class BiometricTemplateGenerator:
    def __init__(self, config_mgr: ConfigManager):
        self.config_mgr = config_mgr
        embedding = self._load_embedding()
        
        if embedding is not None:
            config_mgr.biometric_embedding = embedding
        else:
            config_mgr.biometric_embedding = self.get_new_embedding()

    def get_new_embedding(self) -> np.ndarray:
        encoder = VoiceEncoder()
        audio = self._get_audio()

        assert audio.dtype == np.float32, "Audio must be float32"
        assert audio.ndim == 1, "Audio must be mono (1D ndarray)"

        audio = self._ndarray_to_torch_float32(audio=audio)
        embedding = encoder.embed_utterance(audio)
        # np.save(self.config_mgr.biometric_config.template_path, embedding)
        print(embedding.shape)
        return embedding

    def _ndarray_to_torch_float32(self, audio: np.ndarray) -> Tensor:
        return tensor(audio, dtype=float32)
    
    def _get_audio(self) -> np.ndarray:
        phrase = []
        audio = None
        wd = False

        while len(phrase) < self.config_mgr.biometric_config.audio_sample_required:
            audio = detect_voice(config=self.config_mgr)

            # Temporary until wake_up model is created
            audio_transcript = transcribe_audio(model=self.config_mgr.whisper_model_sm, audio=audio, beam_size=self.config_mgr.model_config.beam_size)
            wd = wad.wake_up_detection_stub(ip=audio_transcript)

            if wd:
                phrase.append(audio)
            
        return np.concatenate(phrase)
            
    def _load_embedding(self) -> np.ndarray:
        try:
            embedding = np.load(self.config_mgr.biometric_config.template_path)
        except (FileNotFoundError, IOError, ValueError) as e:
            return None 

        if not isinstance(embedding, np.ndarray):
            return None

        EMBEDDING_SIZE = 256
        if embedding.shape != (EMBEDDING_SIZE,) or embedding.dtype != np.float32:
            return None

        return embedding