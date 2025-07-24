import numpy as np
from resemblyzer import VoiceEncoder
from config.config_manager import ConfigManager
from torch import tensor, float32, Tensor
from utility import logger
from utility.encrypt import CryptManager
import os

logger = logger.get_logger(__name__)

class BiometricTemplateGenerator:
    def __init__(self, config_mgr: ConfigManager):
        self._config_mgr = config_mgr
        self._crypt_mgr = CryptManager(config=config_mgr)
        self._encoder = VoiceEncoder()
        self._template = self._load_embedding()
        self._is_template = False if len(self._template) == 0 else True

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

        if not embeddings:
            logger.critical("No embeddings generated.")
            raise Exception

        try:
            template = np.mean(np.stack(embeddings), axis=0)
        except Exception as e:
            logger.error(f"Failed to compute template from embeddings: {e}")
            raise
        
        if not isinstance(template, np.ndarray):
            logger.error("Template is not a valid ndarray.")
            raise Exception

        username=input('Enter Username: ')

        self._is_template = True
        self._template[username] = template

        self._save_template(username=username)
    
    def _save_template(self, username: str):
        try:
            filename = os.path.join(self._config_mgr.basic_info.usr_data_dir,
                                    self._config_mgr.biometric_config.get_file_name(username=username))
            
            cipher = self._crypt_mgr.encrypt(self._template[username].tobytes())

            with open(filename, 'wb') as f:
                f.write(cipher)
            logger.info("New biometric template generated and saved.")

        except (OSError, IOError) as e:
            logger.error(f"Failed to save template: {e}")

    def _ndarray_to_torch_float32(self, audio: np.ndarray) -> Tensor:
        return tensor(audio, dtype=float32)
           
    def _load_embedding(self) -> dict[str, np.ndarray]:
        file_list = self._get_template_files(self._config_mgr.basic_info.usr_data_dir)
        dic = {}
        EMBEDDING_SIZE = 256
        for file_name in file_list:
            try:
                with open(os.path.join(self._config_mgr.basic_info.usr_data_dir, file_name), 'rb') as file_content:
                    decrypted = self._crypt_mgr.decrypt(file_content.read())
                    embedding = np.frombuffer(decrypted, dtype=np.float32)
                    embedding = embedding.reshape((EMBEDDING_SIZE,)) 
                    username = self._config_mgr.biometric_config.extract_username(file_name)
                    dic[username] = self._normalize(embedding)
            except (FileNotFoundError, IOError, ValueError) as e:
                return {}

            if not isinstance(embedding, np.ndarray):
                return {}
            
            if embedding.shape != (EMBEDDING_SIZE,) or embedding.dtype != np.float32:
                return {}

        return dic
    
    def match_embedding(self, audio: np.ndarray) -> bool:
        if self._template is None:
            logger.critical("No Embedding Found")
            return False
        
        assert audio.dtype == np.float32, "Audio must be float32"
        assert audio.ndim == 1, "Audio must be mono (1D ndarray)"

        
        new_embedding = self._encoder.embed_utterance(audio)
        new_embedding_norm = self._normalize(new_embedding)

        for username in self._template:
            similarity = np.dot(new_embedding_norm, self._template[username])

            if similarity >= self._config_mgr.biometric_config.threshold:
                logger.info(f'Similarity: {similarity}')
                logger.info(f'Matched with: {username}')
                return True
        
        return False

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Zero-norm embedding; invalid audio or encoder failure.")
        return embedding / norm
    
    def _get_template_files(self, dir_path: str) -> list[str]:
        prefix, suffix = self._config_mgr.biometric_config.get_file_pattern()

        try:
            files = os.listdir(dir_path)
        except FileNotFoundError:
            raise Exception
        except PermissionError:
            raise Exception
        except OSError:
            raise Exception

        return [
            f for f in files
            if f.startswith(prefix) and f.endswith(suffix)
    ]