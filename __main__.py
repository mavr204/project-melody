import core.audio_input_pipeline as ap
from core.VAD import SpeechVAD
from config.input_pipe_config import  AudioConfig, WhisperModelConfig, VADConfig

model_sm = ap.load_model(WhisperModelConfig())
audio_config = AudioConfig()
vad_config = VADConfig(sample_rate=audio_config.sample_rate)

while True:
    audio = ap.record_audio(audio_config)

    check = ap.voice_activity_detector(audio, model_sm, vad_config)
    
    if check == False:
        break
    print('Complete!')