import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from core.input_pipeline import InputPipeline, WakeUpChecks
import utility.errors as err

@pytest.fixture
def mock_config():
    mock_cfg = MagicMock()
    mock_cfg.audio_config.sample_rate = 16000
    mock_cfg.audio_config.channels = 1
    mock_cfg.audio_config.dtype = 'float32'
    mock_cfg.audio_config.duration = 1.0

    mock_cfg.vad_config.frame_duration_ms = 30
    mock_cfg.vad_config.silence_counter_max = 3
    mock_cfg.vad_config.check_wake_after_frames = 3

    mock_cfg.filter_config.normalizing_peak = 0.99
    mock_cfg.filter_config.low_cutoff = 100
    mock_cfg.filter_config.high_cutoff = 3000

    mock_cfg.model_config.model_sm.transcribe.return_value = ([MagicMock(start=0, end=1, text="hello")], MagicMock(language="en", language_probability=0.99))
    mock_cfg.model_config.beam_size = 5

    mock_cfg.biometric_config.audio_sample_required = 1

    return mock_cfg

@pytest.fixture
def mock_template():
    mock_template = MagicMock()
    mock_template.match_embedding.return_value = True
    mock_template.is_template = True
    return mock_template

@pytest.fixture
def pipeline(mock_config, mock_template):
    with patch("core.input_pipeline.ThreadManager") as mock_thread_mgr_class, \
         patch("core.input_pipeline.bandpass_filter", return_value="filter"):
        mock_thread_mgr = MagicMock()
        mock_thread_mgr.active_threads = {'AudioStreamThread': MagicMock(stop_event=MagicMock(is_set=MagicMock(return_value=True)))}
        mock_thread_mgr.create_new_thread.return_value = MagicMock()
        mock_thread_mgr.get_thread_status.return_value = MagicMock()
        mock_thread_mgr.stop_thread = MagicMock()
        mock_thread_mgr.stop_all_threads = MagicMock()

        mock_thread_mgr_class.return_value = mock_thread_mgr

        return InputPipeline(mock_config, mock_template)

def test_byte_to_float32_audio_valid(pipeline):
    raw_bytes = (np.ones(16000, dtype=np.int16)).tobytes()
    result = pipeline.byte_to_float32_audio([raw_bytes])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32

def test_byte_to_float32_audio_invalid(pipeline):
    with pytest.raises(err.InvalidAudioError):
        pipeline.byte_to_float32_audio([])

def test_process_audio_to_bytes(pipeline):
    audio_chunk = np.ones(16000, dtype=np.float32)
    pipeline.queue.put(audio_chunk)
    result = pipeline._process_audio_to_bytes()
    assert isinstance(result, bytes)

def test_process_audio_to_bytes_queue_empty(pipeline):
    with pytest.raises(err.QueueEmptyError, match="Audio queue is empty"):
        pipeline._process_audio_to_bytes()

def test_wake_up_validation(pipeline):
    audio = np.ones(16000, dtype=np.float32)
    checks = WakeUpChecks()
    with patch("core.input_pipeline.wud.wake_up_detection_stub", return_value=True):
        pipeline.wake_up_validation(audio, checks)
    assert checks.wake_up is True
    assert checks.biometric_pass is True

def test_voice_activity_detector_speech_then_silence(pipeline):
    frame_size = 960  # based on your config
    silence_trigger = pipeline._config.vad_config.silence_counter_max

    # Make audio with 5 speech frames, then silence_trigger + 1 silent frames
    total_frames = 5 + silence_trigger + 1
    dummy_audio = b'\x01' * (frame_size * total_frames)  # fake audio bytes

    # Simulate: speech → silence → triggers silence stop
    speech_pattern = [True] * 5 + [False] * (silence_trigger + 1)

    with patch.object(pipeline, "_process_audio_to_bytes", return_value=dummy_audio), \
         patch.object(pipeline.vad, "isSpeech", side_effect=speech_pattern):

        speech_frames = []
        result = pipeline._voice_activity_detector(speech_frames)

        assert result is False
        assert len(speech_frames) == 5

def test_get_command_success(pipeline):
    with patch.object(pipeline, "_record_audio_stream"), \
         patch.object(pipeline, "_wake_up_detect", return_value=np.ones(16000, dtype=np.float32)):
        audio = pipeline.get_command()
    assert isinstance(audio, np.ndarray)

def test_get_command_no_template(pipeline, mock_template):
    mock_template.is_template = False
    pipeline.voice_template = mock_template
    with pytest.raises(err.TemplateLoadError):
        pipeline.get_command()

def test_get_template_audio_success(pipeline):
    with patch.object(pipeline, "_record_audio_stream"), \
         patch.object(pipeline, "byte_to_float32_audio", return_value=np.ones(16000, dtype=np.float32)), \
         patch.object(pipeline, "transcribe_audio", return_value="placeholder for transcription"), \
         patch.object(pipeline, "_voice_activity_detector", side_effect=[False, False]), \
         patch("core.input_pipeline.wud.wake_up_detection_stub", return_value=True):
        
        result = pipeline.get_template_audio()

    assert isinstance(result, list)
    assert all(isinstance(i, np.ndarray) for i in result)
    assert len(result) == pipeline._config.biometric_config.audio_sample_required

