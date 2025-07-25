import numpy as np
from resemblyzer import VoiceEncoder
from config.config_manager import ConfigManager
from utility import logger
from utility.encrypt import CryptManager
import os
import utility.errors as err
from utility.thread_manager import ThreadManager, ThreadStatus

logger = logger.get_logger(__name__)

class BiometricTemplateGenerator:
    _EMBEDDING_SIZE = 256
    _UPDATE_TEMPLATE_THREAD = 'UpdateTemplateThread'
    def __init__(self, config_mgr: ConfigManager):
        self._config_mgr = config_mgr
        self._crypt_mgr = CryptManager(config=config_mgr)
        self.thread_mgr = ThreadManager()

        try:
            self._encoder = VoiceEncoder()
        except Exception as e:
            logger.critical(f"VoiceEncoder init failed: {e}")
            raise err.BiometricError("Failed to initialize encoder")
        
        self._template: dict[str, np.ndarray] = self._load_embedding()
        """
        Waits for any active template update thread to finish.

        Call sync_template_update_thread before accessing templates if consistency is critical.
        """


    @property
    def is_template(self) -> bool:
        return bool(self._template)

    def get_new_template(self, audio_list: list[np.ndarray]) -> None:

        for audio in audio_list:
            try:
                assert audio.dtype == np.float32, "Audio must be float32"
                assert audio.ndim == 1, "Audio must be mono (1D ndarray)"
            except AssertionError as e :
                raise err.BiometricError(str(e)) 

        try:
            embeddings = [
                self._normalize(self._encoder.embed_utterance(audio))
                for audio in audio_list
            ]
        except Exception as e:
            logger.critical(f"Bad Audio:{e} stopping template generation...")
            return

        try:
            template = np.mean(np.stack(embeddings), axis=0)
        except (ValueError, TypeError):
            raise err.TemplateGenerationError("Failed to create template from embeddings")
        
        if not isinstance(template, np.ndarray):
            raise err.TemplateGenerationError("Generated template is not valid")
                
        username = self._get_username()

        self._template[username] = template

        self._save_template(username=username, embedding=template)

    def _get_username(self) -> str:
        while True:
                username=input('Enter Username: ').strip()
                if username in self._template:
                    logger.warning("Template for this username already exists and will be overwritten.")
                if username:
                    return username
                logger.error("Invalid Username! Try again...")
    
    def _save_template(self, username: str, embedding: np.ndarray) -> None:
        for attempt_num in range(3): # Tries to save the template 3 times
            try:
                filename = os.path.join(self._config_mgr.basic_info.usr_data_dir,
                                        self._config_mgr.biometric_config.get_file_name(username=username))
                
                cipher = self._crypt_mgr.encrypt(embedding.tobytes())

                with open(filename, 'wb') as f:
                    f.write(cipher)
                logger.info("New biometric template generated and saved.")

            except (OSError, IOError):
                if attempt_num < 2:
                    logger.error(f"Could not save template. Failed Attempt: {attempt_num+1}. Retrying...")
                else:
                    raise err.FileAccessError(f"Failed write on Disk. After {attempt_num+1}")
           
    def _load_embedding(self) -> dict[str, np.ndarray]:
        file_list = self._get_template_files(self._config_mgr.basic_info.usr_data_dir)
        dic = {}
        for file_name in file_list:
            try:
                with open(os.path.join(self._config_mgr.basic_info.usr_data_dir, file_name), 'rb') as file_content:
                    decrypted = self._crypt_mgr.decrypt(file_content.read())
                    embedding = np.frombuffer(decrypted, dtype=np.float32)
                    embedding = embedding.reshape((self._EMBEDDING_SIZE,)) 
                    username = self._config_mgr.biometric_config.extract_username(file_name)
                    dic[username] = self._normalize(embedding)
            except (FileNotFoundError, IOError, ValueError) as e:
                logger.warning(f"Skipping file {file_name} due to error: {e}")
                continue

        return dic
    
    def match_embedding(self, audio: np.ndarray) -> bool:
        if len(self._template) == 0:
            logger.critical("No Embedding Found")
            return False
        
        try:
            assert audio.dtype == np.float32, "Audio must be float32"
            assert audio.ndim == 1, "Audio must be mono (1D ndarray)"
        except AssertionError as e:
            logger.error(f"Invalid Audio data: {e}")
            return False

        
        try:
            new_embedding = self._encoder.embed_utterance(audio)
            new_embedding_norm = self._normalize(new_embedding)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return False

        for username in self._template:
            similarity = np.dot(new_embedding_norm, self._template[username])

            if similarity >= self._config_mgr.biometric_config.threshold:
                logger.debug(f'Similarity: {similarity}')
                logger.debug(f'Matched with: {username}')
                self.start_template_update_thread(username=username, embedding=new_embedding_norm)
                return True
        
        return False

    def start_template_update_thread(self, username: str, embedding: np.ndarray) -> None:
        self.thread_mgr.create_new_thread(target=self._roll_template_update,
                                          args=(username, embedding),
                                          name=self._UPDATE_TEMPLATE_THREAD,
                                          autostart=True)

    def sync_template_update_thread(self):
        if self.thread_mgr.get_thread_status(self._UPDATE_TEMPLATE_THREAD) == ThreadStatus.RUNNING:
            self.thread_mgr.stop_thread(self._UPDATE_TEMPLATE_THREAD)

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Zero-norm embedding; invalid audio or encoder failure.")
        return embedding / norm
    
    def _get_template_files(self, dir_path: str) -> list[str]:
        prefix, suffix = self._config_mgr.biometric_config.get_file_pattern()

        try:
            files = os.listdir(dir_path)
        except (FileNotFoundError, OSError, PermissionError):
            raise err.FileAccessError("Template Files could not be accessed")

        return [
            f for f in files
            if f.startswith(prefix) and f.endswith(suffix)
    ]

    def _roll_template_update(self, username:str, embedding: np.ndarray) -> None:
        existing_template: np.ndarray = self._template[username]

        new_embedding_weight = self._config_mgr.biometric_config.update_weight
        existing_weight = (1 - new_embedding_weight)
        
        updated_embedding = (existing_weight * existing_template) + (new_embedding_weight * embedding)
        updated_embedding = self._normalize(updated_embedding)

        self._template[username] = updated_embedding
        self._save_template(username=username, embedding=updated_embedding)
