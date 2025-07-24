import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import keyring.errors
from config.config_manager import ConfigManager
from utility.logger import get_logger
import keyring
import utility.errors as err

logger = get_logger(__name__)

class CryptManager:
    # Encrypts data using AES-GCM.
    # Returns: nonce + ciphertext + tag

    def __init__(self, config: ConfigManager) -> None:
        _KEYRING_SERVICE = config.basic_info.app_name
        _KEYRING_USERNAME = "biometric-template-key"
        self._config = config
        try:
            key = keyring.get_password(service_name=_KEYRING_SERVICE, username=_KEYRING_USERNAME)
        except keyring.errors.KeyringError as e:
            logger.critical(f"Keyring access failed: {e}")
            raise err.EncryptionError("Could not access keyring")

        if key is None:
            try:
                key = AESGCM.generate_key(bit_length=config.biometric_config.encrypt_key_length)
                keyring.set_password(
                    service_name=_KEYRING_SERVICE,
                    username=_KEYRING_USERNAME,
                    password=key.hex()
                )
            except Exception as e:
                logger.critical(f"Failed to generate or store encryption key: {e}")
                raise err.EncryptionError("Failed to generate/store encryption key.")
        else:
            try:
                key = bytes.fromhex(key)
            except ValueError as e:
                logger.critical(f"Stored key is invalid: {e}")
                raise err.EncryptionError("Corrupted encryption key in keyring.")

        self._aesgcm = AESGCM(key)
        del key
    
    def encrypt(self, data: bytes) -> bytes:
        try:
            nonce = os.urandom(self._config.biometric_config.nonce_bytes)
            cipher_text = self._aesgcm.encrypt(nonce=nonce, data=data, associated_data=None)
            return nonce + cipher_text
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise err.EncryptionError("Failed to encrypt data.")

    def decrypt(self, data:bytes) -> bytes:
        nonce_size = self._config.biometric_config.nonce_bytes
        try:
            if len(data) < nonce_size:
                raise ValueError("Data is too short to contain valid nonce.")

            nonce = data[:nonce_size]
            cipher_text = data[nonce_size:]
            return self._aesgcm.decrypt(nonce=nonce, data=cipher_text, associated_data=None)
        except ValueError as e:
            logger.error(f"Decryption failed: {e}")
            raise err.EncryptionError(f"Decryption failed: {str(e)}")
        except Exception as e:
            logger.critical(f"Unexpected decryption error: {e}")
            raise err.EncryptionError("Unexpected error during decryption.")
