import core.audio_input_pipeline as ap
from core.VAD import SpeechVAD
from config.input_pipe_config import  AudioConfig, WhisperModelConfig, VADConfig

model_sm = ap.load_model(WhisperModelConfig())
audio_config = AudioConfig()
vad_config = VADConfig(sample_rate=audio_config.sample_rate)


ap.voice_activity_detector(model_sm, vad_config, audio_config)