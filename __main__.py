from config.config_manager import ConfigManager
from core.template_generator import BiometricTemplateGenerator
from core.input_pipeline import InputPipeline
from core.assistant import run_command
from utility import logger, errors as err

logger = logger.get_logger(__name__)

def main():
    config_mgr = ConfigManager()
    biometric_template = BiometricTemplateGenerator(config_mgr=config_mgr)
    audio_input = InputPipeline(config=config_mgr, voice_template=biometric_template)

    while True:
        logger.debug('='*60)
        try:
            audio = audio_input.get_command()
        except err.TemplateLoadError:
            template_audio = audio_input.get_template_audio()
            biometric_template.get_new_template(template_audio)
            del template_audio
            continue

        command = audio_input.transcribe_audio(audio=audio)
        logger.info(command)

        run_command(command=command)
    

if __name__ == "__main__":
    main()

