# Melody

**Melody** is a work-in-progress voice assistant built in Python.
It focuses on privacy-first, local-first voice interaction — no cloud APIs, no telemetry.

---

## Current Capabilities

Melody can currently:

* Record audio from the microphone.
* Detect speech using `webrtcvad`.
* Detect wake word using fuzzy matching (`rapidfuzz`).
* Verify speaker identity via biometric matching using `resemblyzer`.
* Transcribe speech to text using `faster-whisper`.
* Execute simple system commands via `subprocess`.

---

## Features

* ✅ Voice recording (mono WAV, float32)
* ✅ Voice Activity Detection (WebRTC VAD)
* ✅ Wake word detection with fuzzy matching
* ✅ Biometric verification (Resemblyzer)
* ✅ Offline transcription (`faster-whisper`)
* ✅ Modular audio pipeline (dataclass-driven)
* ✅ One-shot command support (e.g., "melody, play music")
* ✅ Persistent biometric template with validation
* ✅ Rolling update of biometric templates on matched attempts
* ✅ Custom Thread management
* ✅ Custom Errors

---

## Requirements

* `Python 3.10+`
* `numpy`
* `scipy`
* `sounddevice`
* `faster-whisper`
* `webrtcvad`
* `resemblyzer`
* `rapidfuzz`
* `torch` (for Resemblyzer)
* `subprocess` (standard library)
* `cryptography`
* `keyring`
* `appdirs`

---

## Roadmap / Planned

* 🎯 **Audio Preprocessing**
  Filter low-frequency noise, suppress background sounds.

* 🎯 **Wake Word Model**
  Replace fuzzy matching with a lightweight neural model (no transcription required).

* 🎯 **Biometric Model Upgrade**
  Move to `ECAPA-TDNN` (e.g., from SpeechBrain) for faster and more robust voice verification.

* 🎯 **NLU / Intent Detection**
  Basic NLU for flexible and natural command interpretation.

* 🎯 **Command Engine Overhaul**
  Move from raw `subprocess` calls to a structured and extensible command execution layer.
