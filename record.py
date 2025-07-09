import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import datetime

def record(duration, sample_rate, channels, d_type, write):
    def save_recording(sample_rate, recording):
        os.makedirs("./samples", exist_ok=True)
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")

        existing_file_names = [f for f in os.listdir("./samples") if f.endswith(".wav") and 'wavf' in f]
        existing_files_serial_numbers = []

        for f in existing_file_names:
            try:
                num = int(f[4:7])
                existing_files_serial_numbers.append(num)
            except:
                pass

        next_num = max(existing_files_serial_numbers, default=0) + 1
        filename = f"./samples/wavf{next_num:03d}_{date_str}.wav"

        
        if recording.dtype == np.float32:
            INT16_MAX = np.iinfo(np.int16).max # Highest value in int16 audio
            wav.write(filename, sample_rate, (recording * INT16_MAX).astype(np.int16))
        else:
            wav.write(filename, sample_rate, recording)

        print(f"Saved as {filename}")

    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype=d_type)
    sd.wait()
    print("Recording finished.")
    recording[:int(0.5 * sample_rate)] = 0 # Removes the noise at the start 0.5 seconds of the recording
    
    if write:
        save_recording(sample_rate, recording)
    
    if channels == 1:   # If the Audio is Mono
        recording = recording.flatten()
    print(recording.shape, recording.dtype, np.max(recording), np.min(recording))
    return recording
