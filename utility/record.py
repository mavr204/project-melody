import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, sosfilt
import os
import datetime

def record(duration, sample_rate, write) -> np.ndarray:
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")
    recording[:int(0.5 * sample_rate)] = 0

    recording = recording.flatten()
    print("Non-Filtered:\ndtype: ", recording.dtype, "ndim: ", recording.ndim)
    
    if write:
        save_recording(sample_rate, recording)
    return recording

def save_recording(sample_rate, recording, filename: str = ''):
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
    
    if filename == '':
        
        filename = f"./samples/wavf{next_num:03d}_{date_str}.wav"
    else:
        next_num = int(next_num-1)
        filename = f"./samples/{filename}{next_num:03d}_{date_str}.wav"

    if recording.ndim == 1:
        recording = recording.reshape(-1, 1)
    
    if recording.dtype == np.float32:
        recording = np.int16(recording * 32767)
    
    wav.write(filename, sample_rate, recording)
    print(f"Saved as {filename}")

