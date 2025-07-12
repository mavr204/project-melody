from rapidfuzz import fuzz
from core.audio_input_pipeline import record_audio, transcribe_audio
from config.config_manager import ConfigManager
import numpy as np

config_mgr = ConfigManager()
def fuzz_test():
    score = None
    ip = None
    wake_up_phrases = ['melody', 'Hi melody', 'yo melody']

    while True:
        ip:str = input('Enter phrase: ')
        for phrase in wake_up_phrases:
            score = fuzz.ratio(ip, phrase)
            if score >= 80:
                print(f'Passed, Score: {score}%, phrase: "{phrase}"')

def record_and_transcribe():
    print(transcribe_audio(config_mgr.whisper_model_sm, record_audio().flatten(), config_mgr.model_config.beam_size))

record_and_transcribe()
