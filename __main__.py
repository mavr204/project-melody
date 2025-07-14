from config.config_manager import ConfigManager
from core.audio_input_pipeline import detect_voice, transcribe_audio
from core.template_generator import BiometricTemplateGenerator
import stubs.wake_up_detection as wud
import time

def main():
    config_mgr = ConfigManager()
    biometric_template = BiometricTemplateGenerator(config_mgr=config_mgr)

    # while True:
    print('==================================================================')
    voice = detect_voice(config=config_mgr)

    start = time.time()
    # if wud.wake_up_detection_stub(ip=transcript) and biometric_template.match_embedding(voice):
    if biometric_template.match_embedding(voice):
        print(f"[+{time.time() - start:.2f}s] Biometric passed")
        transcript = transcribe_audio(config=config_mgr, audio=voice)
        print(transcript)
    else:
        print("Failed Auth or Failed Wake Up")
        # continue

    print('fck')
    

if __name__ == "__main__":
    main()

