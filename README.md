# Melody

**Melody** is a work-in-progress voice assistant built with Python.

So far, it can:
- Record audio
- Detect speech using Voice Activity Detection (VAD) using `webrtcvad`
- Process the audio using `rapidfuzz` to detect wakeup word
- Biometric checking with `resemblyzer` to verify user
- Transcribe speech to text using `faster-whisper`
- Using `subprocess` to run the command

---

## Features
- ✅ Voice recording (mono, WAV)
- ✅ Voice Activity Detection (WebRTC VAD)
- ✅ Speech-to-text transcription (faster-whisper)
- ✅ Configurable audio pipeline using dataclasses
- ✅ Wake up word detection using partial fuzzy matching
- ✅ Biometric verification using `resemblyzer`

---

## Requirements
- `Python 3.10+`
- `sounddevice`
- `numpy`
- `scipy`
- `faster-whisper`
- `webrtcvad`
- `resemblyzer`
- `rapidfuzz`
- `subprocess`

---

## Future
- Add better audio processing and parallel processing, for better user experience
- Change `resemblyzer` to something faster, possibly an `ECAPA-TDNN` model. Maybe from speechbrain.
- Build a custom model for wake up word detection, so transcription is not needed for the wake word. Improving performance.
- Build intent detection, possibly using NLU. To improve seemless conversation.
- Better command processing.
