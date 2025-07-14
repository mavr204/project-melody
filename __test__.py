from rapidfuzz import fuzz
from core.audio_input_pipeline import transcribe_audio, detect_voice
from config.config_manager import ConfigManager
from core.template_generator import BiometricTemplateGenerator
import numpy as np
import stubs.wake_up_detection as wad
from resemblyzer import preprocess_wav

config_mgr = ConfigManager()

def template_gen():
    biometric_template = BiometricTemplateGenerator(config_mgr=config_mgr)
    audio = preprocess_wav("./samples/wavf001_2025-07-07.wav")

    match = biometric_template.match_embedding(audio=audio)
    print(match)


template_gen()