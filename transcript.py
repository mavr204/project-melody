from faster_whisper import WhisperModel
from record import record
import threading

model = None # For the model after it has been created
model_size = "small"
rec = None # For recorded audio


def load_model():
    global model
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print('Model Loaded.')
def record_audio():
    global rec
    rec = record(5, 16_000, 1, 'float32', False)

# load the model and record on separate threads
thread_model = threading.Thread(target=load_model)
thread_record = threading.Thread(target=record_audio)
# Start the threads
thread_model.start()
thread_record.start()
# Wait until both the model and the audio is ready for transcription
thread_model.join()
thread_record.join()


# segments, info = model.transcribe("./samples/wavf001_2025-07-07.wav", beam_size=5)
segments, info = model.transcribe(rec, beam_size=5)


print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))