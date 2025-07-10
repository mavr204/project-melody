import webrtcvad
from config.input_pipe_config import VADConfig

class SpeechVAD:

    def __init__(self, config:VADConfig):
        self.config = config
        self.vad = webrtcvad.Vad(config.aggressiveness)
        print("Aggressiveness: ",config.aggressiveness)

    def isSpeech(self, frame:bytes)->bool:
        """Listening for Speech"""
        return self.vad.is_speech(frame,self.config.sample_rate)