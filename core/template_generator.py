import numpy as np
from resemblyzer import VoiceEncoder
# from core.audio_input_pipeline import transcribe_audio, detect_voice
from config.config_manager import ConfigManager
import stubs.wake_up_detection as wad
from torch import tensor, float32, Tensor
from utility import logger

logger = logger.get_logger(__name__)

class BiometricTemplateGenerator:
    _config_mgr = None
    _encoder = None
    _template = None

    def __init__(self, config_mgr: ConfigManager, gen_template: bool = False):
        self._config_mgr = config_mgr
        self._encoder = VoiceEncoder()
        embedding = self._load_embedding()
        
        if embedding is not None and not gen_template:
            embedding = self._normalize(embedding=embedding)
            self._template = embedding
        else:
            ("generating new embedding...")
            embedding = self._get_new_embedding()
            self._template = self._normalize(embedding)

    def _get_new_embedding(self) -> np.ndarray:
        
        audio = self._get_audio()

        assert audio.dtype == np.float32, "Audio must be float32"
        assert audio.ndim == 1, "Audio must be mono (1D ndarray)"

        # audio = self._ndarray_to_torch_float32(audio=audio)
        embedding = self._encoder.embed_utterance(audio)
        np.save('./template/template_audio.npy', audio)
        np.save(self._config_mgr.biometric_config.template_path, embedding)

        return embedding

    def _ndarray_to_torch_float32(self, audio: np.ndarray) -> Tensor:
        return tensor(audio, dtype=float32)
    
    def _get_audio(self) -> np.ndarray:
        phrase = []
        audio = None
        wd = False

        # for _ in range(self._config_mgr.biometric_config.audio_sample_required):
        #     audio = detect_voice(config=self._config_mgr)

        #     # Temporary until wake_up model is created
        #     audio_transcript = transcribe_audio(config=self._config_mgr, audio=audio)
        #     wd = wad.wake_up_detection_stub(ip=audio_transcript)

        #     if wd:
        #         phrase.append(audio)
            
        return np.concatenate(phrase)
            
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

        return embedding
    
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