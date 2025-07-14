import numpy as np
from resemblyzer import VoiceEncoder
from torch import tensor, float32, Tensor

class BiometricTemplateGenerator:
    def get_new_embedding(self, audio: np.ndarray) -> np.ndarray:
        encoder = VoiceEncoder()

        assert audio.dtype == np.float32
        assert audio.ndim == 1

        audio = self._ndarray_to_torch_float32(audio=audio)
        embedding = encoder.embed_utterance(audio)
        # np.save("./template/voice_template.npy", embedding)
        return embedding

    def _ndarray_to_torch_float32(self, audio: np.ndarray) -> Tensor:
        return tensor(audio, dtype=float32)