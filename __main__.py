from config.config_manager import ConfigManager
from core.audio_input_pipeline import detect_voice, transcribe_audio
from core.template_generator import BiometricTemplateGenerator
import stubs.wake_up_detection as wud
from core.assistant import run_command

def main():
    config_mgr = ConfigManager()
    biometric_template = BiometricTemplateGenerator(config_mgr=config_mgr)

    while True:
        print('==================================================================')
        voice = detect_voice(config=config_mgr)

        transcript = transcribe_audio(config=config_mgr, audio=voice)
        if wud.wake_up_detection_stub(ip=transcript) and biometric_template.match_embedding(voice):
        # if biometric_template.match_embedding(voice):
            
            print("Listening for command...")
            voice = detect_voice(config=config_mgr)
            transcript = transcribe_audio(config=config_mgr, audio=voice)
            print(f"Command: {transcript}")
            run_command(transcript)

        else:
            print("Failed Auth or Failed Wake Up")
            continue
    

if __name__ == "__main__":
    main()

