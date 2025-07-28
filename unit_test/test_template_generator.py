
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from core.template_generator import BiometricTemplateGenerator

@pytest.fixture
def mock_config_mgr():
    mock_cfg = MagicMock()
    mock_cfg.basic_info.usr_data_dir = "/tmp"
    mock_cfg.biometric_config.get_file_name.return_value = "test_user.biotemplate"
    mock_cfg.biometric_config.get_file_pattern.return_value = ("", ".biotemplate")
    mock_cfg.biometric_config.extract_username.return_value = "test_user"
    mock_cfg.biometric_config.threshold = 0.7
    mock_cfg.biometric_config.update_weight = 0.5
    return mock_cfg

@pytest.fixture
def mock_embedding():
    return np.ones(256, dtype=np.float32)

@pytest.fixture
def audio_list():
    return [np.ones(256, dtype=np.float32) for _ in range(3)]

@patch("core.template_generator.os.listdir", return_value=["test_user.biotemplate"])
@patch("core.template_generator.CryptManager")
@patch("core.template_generator.ThreadManager")
@patch("core.template_generator.logger")
@patch("core.template_generator.VoiceEncoder")
def test_get_new_template(mock_encoder_cls, mock_logger, mock_thread_mgr, mock_crypt_mgr, mock_listdir,
                          mock_config_mgr, mock_embedding, audio_list):

    mock_encoder = mock_encoder_cls.return_value
    mock_encoder.embed_utterance.return_value = mock_embedding

    mock_crypt = mock_crypt_mgr.return_value
    mock_crypt.decrypt.return_value = mock_embedding.tobytes()
    mock_crypt.encrypt.return_value = b"encrypted_data"

    generator = BiometricTemplateGenerator(mock_config_mgr)

    with patch.object(generator, "_get_username", return_value="test_user"), patch.object(generator, "_save_template") as mock_save:

        generator.get_new_template(audio_list)

        assert "test_user" in generator._template
        template = generator._template["test_user"]
        assert isinstance(template, np.ndarray)
        assert template.shape == (256,)
        assert np.isclose(np.linalg.norm(template), 1.0)

        mock_save.assert_called_once_with(username="test_user", embedding=template)


# def test_get_username()