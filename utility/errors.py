class BiometricError(Exception):
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

class KeyringAccessError(EncryptionError):
    def __init__(self, *args):
        super().__init__(*args)