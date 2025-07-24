class BiometricError(Exception):
    def __init__(self, *args):
        super().__init__(*args)

class InputPipelineError(Exception):
    def __init__(self, *args):
        super().__init__(*args)

class InvalidAudioError(InputPipelineError):
    def __init__(self, *args):
        super().__init__(*args)

class ModelLoadError(InputPipelineError):
    def __init__(self, *args):
        super().__init__(*args)

class AudioStreamError(InputPipelineError):
    def __init__(self, *args):
        super().__init__(*args)

class TranscriptionError(InputPipelineError):
    def __init__(self, *args):
        super().__init__(*args)

class FileAccessError(Exception):
    def __init__(self, *args):
        super().__init__(*args)

class EncryptionError(BiometricError):
    def __init__(self, *args):
        super().__init__(*args)
    
class TemplateGenerationError(BiometricError):
    def __init__(self, *args):
        super().__init__(*args)

class TemplateLoadError(BiometricError):
    def __init__(self, *args):
        super().__init__(*args)

class KeyringAccessError(EncryptionError):
    def __init__(self, *args):
        super().__init__(*args)