from rapidfuzz import fuzz
from core.audio_input_pipeline import transcribe_audio, detect_voice
from config.config_manager import ConfigManager
from core.template_generator import BiometricTemplateGenerator
import numpy as np
import stubs.wake_up_detection as wad

config_mgr = ConfigManager()

def template_gen():
    phrase = []
    audio = None
    wd = False
    for i in range(3):
        audio = detect_voice(config=config_mgr)
        audio_transcript = transcribe_audio(model=config_mgr.whisper_model_sm, audio=audio, beam_size=config_mgr.model_config.beam_size)
        wd = wad.wake_up_detection_stub(ip=audio_transcript)

        print('Audio Transcript: ' + audio_transcript, '\nWake Up Detection: ', wd)
        if wd:
            phrase.append(audio)
        
    full_audio = np.concatenate(phrase)
    audio_transcript = transcribe_audio(model=config_mgr.whisper_model_sm, audio=full_audio, beam_size=config_mgr.model_config.beam_size)
    print('Audio Transcript: ' + audio_transcript)

    biometric_template = BiometricTemplateGenerator()
    template = biometric_template.get_new_embedding(full_audio)
    print(template)


template_gen()