import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import keyring.errors
from config.config_manager import ConfigManager
from utility.logger import get_logger
import keyring

logger = get_logger(__name__)

class CryptManager:
    # Encrypts data using AES-GCM.
    # Returns: nonce + ciphertext + tag

    def __init__(self, config: ConfigManager):
        _KEYRING_SERVICE = config.basic_info.app_name
        _KEYRING_USERNAME = "biometric-template-key"
        self._config = config
        try:
            key = keyring.get_password(service_name=_KEYRING_SERVICE, username=_KEYRING_USERNAME)
        except keyring.errors.KeyringError as e:
            logger.critical(f"Keyring access failed: {e}")
            raise SystemExit(1)

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
                raise SystemExit(1)
        else:
            try:
                key = bytes.fromhex(key)
            except ValueError as e:
                logger.critical(f"Stored key is invalid: {e}")
                raise SystemExit(1)

        self._aesgcm = AESGCM(key)
        del key
    
    def encrypt(self, data: bytes) -> bytes:
        nonce = os.urandom(self._config.biometric_config.nonce_bytes)
        cipher_text = self._aesgcm.encrypt(nonce=nonce, data=data, associated_data=None)

        return nonce + cipher_text

    def decrypt(self, data:bytes) -> bytes:
        nonce = data[:self._config.biometric_config.nonce_bytes]
        cipher_text = data[self._config.biometric_config.nonce_bytes:]

        return self._aesgcm.decrypt(nonce=nonce, data=cipher_text, associated_data=None)
