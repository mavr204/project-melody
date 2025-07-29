import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from core.template_generator import BiometricTemplateGenerator
import builtins
from io import BytesIO

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

@pytest.fixture
def mock_encoder(mock_embedding):
    mock_encoder = MagicMock()
    mock_encoder.embed_utterance.return_value = mock_embedding

    return mock_encoder

@pytest.fixture
def mock_thread_mgr():
    mock_thread_mgr = MagicMock()
    return mock_thread_mgr

@pytest.fixture
def mock_crypt_mgr(mock_embedding):
    mock_crypt_mgr = MagicMock()
    mock_crypt_mgr.decrypt.return_value = mock_embedding.tobytes()
    mock_crypt_mgr.encrypt.return_value = b"encrypted_data"
    
    return mock_crypt_mgr

@pytest.fixture
def bt(mock_config_mgr, mock_encoder, mock_crypt_mgr, mock_thread_mgr):
    with patch("core.template_generator.VoiceEncoder", return_value=mock_encoder), \
         patch("core.template_generator.CryptManager", return_value=mock_crypt_mgr), \
         patch("core.template_generator.ThreadManager", return_value=mock_thread_mgr):
        return BiometricTemplateGenerator(mock_config_mgr)
    

@patch("core.template_generator.os.listdir", return_value=["test_user.biotemplate"])
@patch("core.template_generator.logger")
def test_get_new_template(mock_logger, mock_listdir, bt, audio_list):
    generator = bt

    with patch.object(generator, "_get_username", return_value="test_user"), patch.object(generator, "_save_template") as mock_save:

        generator.get_new_template(audio_list)

        assert "test_user" in generator._template
        template = generator._template["test_user"]
        assert isinstance(template, np.ndarray)
        assert template.shape == (256,)
        assert np.isclose(np.linalg.norm(template), 1.0)

        mock_save.assert_called_once_with(username="test_user", embedding=template)

def test_get_username_valid(bt, caplog):
    generator = bt

    with patch("builtins.input", return_value="test_user"):
        username = generator._get_username()
        assert isinstance(username, str)
        assert username == "test_user"

def test_get_username_overwrite_warn(bt, caplog):
    generator = bt
    generator._template["test_user"] = np.ones(256) 

    with caplog.at_level("WARNING"):
        with patch("builtins.input", return_value="test_user"):
            username = generator._get_username()
            assert username == "test_user"
            assert "already exists" in caplog.text

def test_get_username_retry_on_empty(bt, caplog):
    generator = bt
    inputs = ["", " ", "\t", "test_user"]

    with caplog.at_level("ERROR"):
        with patch("builtins.input", side_effect=inputs):
            username = generator._get_username()
            assert username == "test_user"
            assert caplog.text.count("Invalid Username") >= 1

@patch("core.template_generator.logger")
def test_load_embedding(mock_logger, bt, mock_embedding):
    generator = bt

    mock_files = ["file1.biotemplate", "file2.biotemplate"]
    fake_file_data = mock_embedding.tobytes()

    with patch.object(generator, "_get_template_files", return_value=mock_files), \
         patch.object(generator, "_normalize", return_value=mock_embedding), \
         patch("builtins.open", return_value=BytesIO(fake_file_data)) as mock_open:

        loaded = generator._load_embedding()

        assert "test_user" in loaded
        assert isinstance(loaded["test_user"], np.ndarray)
        assert np.allclose(loaded["test_user"], mock_embedding)

        assert mock_open.call_count == len(mock_files)

@patch("core.template_generator.open", new_callable=MagicMock)
def test_save_template_success(mock_open, bt, mock_embedding):
    generator = bt

    with patch.object(generator._crypt_mgr, "encrypt", return_value=b"encrypted_data"):
        generator._save_template("test_user", mock_embedding)

        mock_open.assert_called_once()
        handle = mock_open.return_value.__enter__.return_value
        handle.write.assert_called_once_with(b"encrypted_data")

@patch("core.template_generator.logger")
@patch("core.template_generator.open", side_effect=OSError("Disk error"))
def test_save_template_failure(mock_open, mock_logger, bt, mock_embedding):
    generator = bt

    with pytest.raises(Exception) as exc_info:
        generator._save_template("test_user", mock_embedding)

    assert "Failed write on Disk" in str(exc_info.value)
    assert mock_open.call_count == 3

def test_match_embedding_success(bt, mock_embedding):
    generator = bt
    generator._template["test_user"] = mock_embedding  # already normalized
    with patch.object(generator._encoder, "embed_utterance", return_value=mock_embedding), \
         patch.object(generator, "start_template_update_thread") as mock_thread:

        audio = np.ones(256, dtype=np.float32)
        assert generator.match_embedding(audio) is True
        mock_thread.assert_called_once()

def test_match_embedding_no_match(bt, mock_embedding):
    generator = bt
    generator._template["test_user"] = np.zeros(256, dtype=np.float32)
    with patch.object(generator._encoder, "embed_utterance", return_value=mock_embedding):
        audio = np.ones(256, dtype=np.float32)
        assert generator.match_embedding(audio) is False

def test_normalize_zero_vector(bt):
    generator = bt
    with pytest.raises(ValueError):
        generator._normalize(np.zeros(256, dtype=np.float32))

@patch("core.template_generator.os.listdir", return_value=["a.biotemplate", "b.txt", "c.biotemplate"])
def test_get_template_files(mock_listdir, bt):
    generator = bt
    files = generator._get_template_files("/tmp")
    assert files == ["a.biotemplate", "c.biotemplate"]

@patch("core.template_generator.os.remove")
@patch("core.template_generator.logger")
def test_delete_all_templates(mock_logger, mock_remove, bt):
    generator = bt
    with patch.object(generator, "_get_template_files", return_value=["f1.biotemplate", "f2.biotemplate"]):
        generator._delete_all_templates()
        assert mock_remove.call_count == 2
