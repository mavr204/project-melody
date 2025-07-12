from config.config_manager import ConfigManager
from core.audio_input_pipeline import detect_voice

def main():
    config_mgr = ConfigManager()
    
    detected = detect_voice(
        model=config_mgr.whisper_model_sm,
        vad_config=config_mgr.vad_config,
        audio_config=config_mgr.audio_config,
        beam_size=config_mgr.model_config.beam_size
    )
    print("Detection:", detected)

if __name__ == "__main__":
    main()

