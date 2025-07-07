from record import record
import subprocess
import os

def transcribe(input, output_path, threads=12):
    """Transcribe audio using whisper.cpp."""
    model = './models/ggml-small.en.bin'
    cmd = [
        "./whisper.cpp/build/bin/whisper-cli",
        "-m", model,
        "-f", input,
        "-t", str(threads),
        "-otxt",
        "-of", output_path
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: whisper.cpp transcription failed with return code {e.returncode}")
        return None
    except FileNotFoundError:
        print("Error: whisper-cli binary not found. Check the path.")
        return None

    output_path = output_path + '.txt'

    try:
        with open(output_path, 'r') as f:
            transcript = f.read()
    except FileNotFoundError:
        print(f"Error: Transcript file not found: {output_path}")
        transcript = None
    except IOError as e:
        print(f"IO error reading transcript: {e}")
        transcript = None
    else:
        os.remove(output_path) 
    return transcript


def gen_txt_filename():
    os.makedirs("./samples_txt_op", exist_ok=True)
    existing = [f for f in os.listdir("./samples_txt_op") if f.endswith(".txt") and 'opf' in f]
    numbers = []

    for f in existing:
        try:
            num = int(f[3:6])
            numbers.append(num)
        except:
            pass

    next_num = max(numbers, default=0) + 1
    return f"./samples_txt_op/opf{next_num:0d}"

filename = "./samples/wavf001_2025-07-07.wav"
transcript = transcribe(filename, gen_txt_filename())
print("Transcription:\n", transcript)
print(gen_txt_filename())

