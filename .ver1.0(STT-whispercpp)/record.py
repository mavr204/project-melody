import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import datetime

def record(duration, sample_rate, write):
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
    sd.wait()
    print("Recording finished.")
    recording[:int(0.5 * sample_rate)] = 0
    
    if write:
        save_recording(sample_rate, recording)
    return recording

def save_recording(sample_rate, recording):
    os.makedirs("./samples", exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    existing = [f for f in os.listdir("./samples") if f.endswith(".wav") and 'wavf' in f]
    numbers = []

    for f in existing:
        try:
            num = int(f[4:7])
            numbers.append(num)
        except:
            pass

    next_num = max(numbers, default=0) + 1
    filename = f"./samples/wavf{next_num:03d}_{date_str}.wav"
    
    wav.write(filename, sample_rate, recording)
    print(f"Saved as {filename}")

record(5, 44_100, True)