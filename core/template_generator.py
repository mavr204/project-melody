import numpy as np
from resemblyzer import VoiceEncoder
from config.config_manager import ConfigManager
from torch import tensor, float32, Tensor
from utility import logger

logger = logger.get_logger(__name__)

class BiometricTemplateGenerator:
    def __init__(self, config_mgr: ConfigManager):
        self._config_mgr = config_mgr
        self._encoder = VoiceEncoder()
        self._template = self._load_embedding()
        self._is_template = False if self._template is None else True

    @property
    def is_template(self) -> bool:
        return self._is_template

    def get_new_template(self, audio_list: list[np.ndarray]) -> None:

        for audio in audio_list:
            assert audio.dtype == np.float32, "Audio must be float32"
            assert audio.ndim == 1, "Audio must be mono (1D ndarray)"

        embeddings = [
            self._normalize(self._encoder.embed_utterance(audio))
            for audio in audio_list
        ]

        
        self._template = np.mean(np.stack(embeddings), axis=0)
        self._is_template = True

        try:
            np.save(self._config_mgr.biometric_config.template_path, self._template)
            logger.info("New biometric template generated and saved.")
        except (OSError, IOError) as e:
            logger.error(f"Failed to save template: {e}")

    def _ndarray_to_torch_float32(self, audio: np.ndarray) -> Tensor:
        return tensor(audio, dtype=float32)
           
    def _load_embedding(self) -> np.ndarray:
        try:
            embedding = np.load(self._config_mgr.biometric_config.template_path)
        except (FileNotFoundError, IOError, ValueError) as e:
            return None 

        if not isinstance(embedding, np.ndarray):
            return None

        EMBEDDING_SIZE = 256
        if embedding.shape != (EMBEDDING_SIZE,) or embedding.dtype != np.float32:
            return None

        return self._normalize(embedding)
    
    def match_embedding(self, audio: np.ndarray) -> bool:
        if self._template is None:
            logger.critical("No Embedding Found")
            return False
        
        assert audio.dtype == np.float32, "Audio must be float32"
        assert audio.ndim == 1, "Audio must be mono (1D ndarray)"

        # audio = self._ndarray_to_torch_float32(audio=audio)
        new_embedding = self._encoder.embed_utterance(audio)
        new_embedding_norm = self._normalize(new_embedding)

        similarity = np.dot(new_embedding_norm, self._template)

        logger.info(f'Similarity: {similarity}')
        return similarity >= self._config_mgr.biometric_config.threshold

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Zero-norm embedding; invalid audio or encoder failure.")
        return embedding / norm