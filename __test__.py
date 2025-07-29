from core.template_generator import BiometricTemplateGenerator
from config.config_manager import ConfigManager
from core.input_pipeline import InputPipeline

config = ConfigManager()

bt = BiometricTemplateGenerator(config_mgr=config)

for username in bt._template:
    print(username)