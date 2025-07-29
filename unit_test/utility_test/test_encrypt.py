import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from unittest.mock import patch, MagicMock
from utility.encrypt import CryptManager
import utility.errors as err
import keyring.errors


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.basic_info.app_name = "test-app"
    cfg.biometric_config.encrypt_key_length = 128
    cfg.biometric_config.nonce_bytes = 12
    return cfg

def test_key_is_generated_and_stored(mock_config):
    with patch("utility.encrypt.keyring") as mock_keyring:
        mock_keyring.get_password.return_value = None

        CryptManager(mock_config)

        mock_keyring.set_password.assert_called_once()
        mock_keyring.get_password.assert_called_once()

def test_key_is_retrieved_if_exists(mock_config):
    with patch("utility.encrypt.keyring") as mock_keyring:
        key = AESGCM.generate_key(128).hex()
        mock_keyring.get_password.return_value = key

        CryptManager(mock_config)

        mock_keyring.get_password.assert_called_once()
        mock_keyring.set_password.assert_not_called()

def test_raises_if_key_is_corrupt(mock_config):
    with patch("utility.encrypt.keyring") as mock_keyring:
        mock_keyring.get_password.return_value = "nothex"

        with pytest.raises(err.EncryptionError):
            CryptManager(mock_config)

def test_raises_if_keyring_fails(mock_config):
    with patch("utility.encrypt.keyring.get_password", side_effect=keyring.errors.KeyringError("fail")):
        with pytest.raises(err.EncryptionError):
            CryptManager(mock_config)

def test_encrypt_decrypt_cycle(mock_config):
    with patch("utility.encrypt.keyring") as mock_keyring:
        key = AESGCM.generate_key(128)
        mock_keyring.get_password.return_value = key.hex()

        cm = CryptManager(mock_config)

        data = b"melody"
        enc = cm.encrypt(data)
        dec = cm.decrypt(enc)

        assert dec == data

def test_encrypt_raises(mock_config):
    with patch("utility.encrypt.keyring.get_password", return_value=AESGCM.generate_key(128).hex()):
        with patch("utility.encrypt.AESGCM") as mock_aesgcm_class:
            instance = mock_aesgcm_class.return_value
            instance.encrypt.side_effect = Exception("fail")

            cm = CryptManager(mock_config)
            with pytest.raises(err.EncryptionError):
                cm.encrypt(b"data")

def test_decrypt_short_data(mock_config):
    with patch("utility.encrypt.keyring") as mock_keyring:
        key = AESGCM.generate_key(128)
        mock_keyring.get_password.return_value = key.hex()

        cm = CryptManager(mock_config)

        with pytest.raises(err.EncryptionError):
            cm.decrypt(b"short")

def test_decrypt_fails_with_exception(mock_config):
    with patch("utility.encrypt.keyring.get_password", return_value=AESGCM.generate_key(128).hex()):
        with patch("utility.encrypt.AESGCM") as mock_aesgcm_class:
            instance = mock_aesgcm_class.return_value
            instance.encrypt.return_value = b"\x00" * 20  # dummy encrypted data
            instance.decrypt.side_effect = Exception("fail")

            cm = CryptManager(mock_config)
            with pytest.raises(err.EncryptionError):
                cm.decrypt(b"\x00" * 20)
