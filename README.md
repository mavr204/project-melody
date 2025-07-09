# Melody

**Melody** is a work-in-progress voice assistant built with Python.

So far, it can:
- Record audio
- Detect speech using Voice Activity Detection (VAD) using `webrtcvad`
- Transcribe speech to text using `faster-whisper`

---

## Features
- ✅ Voice recording (mono, WAV)
- ✅ Voice Activity Detection (WebRTC VAD)
- ✅ Speech-to-text transcription (faster-whisper)
- ✅ Configurable audio pipeline using dataclasses

---

## Requirements
- `Python 3.10+`
- `sounddevice`
- `numpy`
- `scipy`
- `faster-whisper`
- `webrtcvad`
