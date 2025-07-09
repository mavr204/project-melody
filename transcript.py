from faster_whisper import WhisperModel
from record import record
import threading

def transcribe_live():
    model = None
    model_size = "small"
    rec = None # For recorded audio


    def load_model():
        global model
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print('Model Loaded.')
    def record_audio():
        global rec
        rec = record(duration=20, sample_rate=16_000, channels=1, d_type='float32', write=False)

    thread_record = threading.Thread(target=record_audio)
    thread_model = threading.Thread(target=load_model)

    thread_model.start()
    thread_record.start()

    thread_model.join()
    thread_record.join()

    segments, info = model.transcribe(audio=rec, beam_size=5)


    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    transcribed_text=''

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        transcribed_text += segment.text
    return transcribed_text
