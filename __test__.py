from rapidfuzz import fuzz
from core.audio_input_pipeline import transcribe_audio, detect_voice
from config.config_manager import ConfigManager
from core.template_generator import BiometricTemplateGenerator
import numpy as np
import stubs.wake_up_detection as wad

config_mgr = ConfigManager()

def template_gen():
    biometric_template = BiometricTemplateGenerator(config_mgr=config_mgr)
    template = biometric_template.get_new_embedding()
    print(template)


template_gen()