from config.config_manager import ConfigManager
from core.template_generator import BiometricTemplateGenerator
from core.assistant import run_command

def main():
    config_mgr = ConfigManager()
    biometric_template = BiometricTemplateGenerator(config_mgr=config_mgr)

    while True:
        print('==================================================================')
    

if __name__ == "__main__":
    main()

