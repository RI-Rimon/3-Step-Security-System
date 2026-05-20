3-Layer Biometric Security System

A terminal-based security access system built in Python that uses three simultaneous verification layers to authenticate a user. All three must pass at the same time for access to be granted.

---

## How It Works

When the program runs, the user is asked for camera permission. Once granted, a single camera window opens and simultaneously verifies:

| Layer | Method | What it checks |
|---|---|---|
| Face Recognition | DeepFace (VGG-Face) | Is this the registered user's face? |
| Liveness Detection | Haar Cascade blink detection | Is this a real person, not a photo? |
| Voice Password + Speaker Verification | Vosk + spectral fingerprinting | Did the right person say the right word? |

All three must pass within the time limit. If any one fails, access is denied.

---

## Motivation

I wanted to explore what a lightweight multi-factor biometric authentication system would look like using only a webcam and microphone — no expensive hardware, no cloud APIs.

The challenge I set myself: face recognition alone is easy to spoof with a photo. Voice password alone can be copied. So the system needed to verify all three simultaneously, making it significantly harder to bypass.

---

## Project Structure

```
3 Step Security System/
│
├── zepto_v2.py              # Main security system
├── enroll_voice.py          # Run once to register your voice
│
├── known_faces/
│   └── rimon1.jpg           # Registered face image
│
├── vosk-model-small-en-us-0.15/   # Vosk speech recognition model
├── voice_profile.npy        # Generated after running enroll_voice.py
└── access_log.txt           # Auto-generated access log (last 100 records)
```

---

## Setup

### 1. Install dependencies

```bash
pip install opencv-python pyaudio vosk deepface tf-keras numpy scipy pyttsx3
```

> **Windows users:** If `dlib` fails to install, use the prebuilt wheel:
> ```bash
> pip install https://github.com/sachadee/Dlib/raw/main/dlib-20.0.0-cp313-cp313-win_amd64.whl
> ```

### 2. Download the Vosk model

Download `vosk-model-small-en-us-0.15` from https://alphacephei.com/vosk/models and place the extracted folder in the project directory.

### 3. Add your face image

Place a clear, well-lit photo of your face in the `known_faces/` folder. Update the `KNOWN_FACE_PATH` in `zepto_v2.py` if needed.

### 4. Enroll your voice

```bash
python enroll_voice.py
```

You will say the password word 3 times. A `voice_profile.npy` file will be generated.

### 5. Run the system

```bash
python zepto_v2.py
```

---

## Configuration

All settings are at the top of `zepto_v2.py`:

```python
VOICE_PASSWORD           = "hello"    # Word to say as password
FACE_TOLERANCE           = 0.5        # Face match strictness (lower = stricter)
VOICE_MATCH_THRESHOLD    = 0.82       # Speaker similarity threshold (0.0–1.0)
SCAN_TIMEOUT             = 25         # Seconds to complete verification
MAX_ATTEMPTS             = 3          # Attempts before lockout
```

---

## Access Log

Every access attempt is logged to `access_log.txt` with a timestamp:

```
[2026-05-17 10:30:15] GRANTED — face, blink, voice and speaker verified
[2026-05-17 10:35:22] DENIED — attempt 1 failed
[2026-05-17 10:36:10] CANCELLED — user denied camera access
```

The log automatically keeps only the last 100 records.

---

## Known Limitations

- **Speaker verification** uses a lightweight spectral fingerprinting approach, not a dedicated speaker embedding model. It is reasonably effective but not production-grade.
- **Liveness detection** uses eye blink detection via Haar cascades. A sophisticated 3D mask could potentially bypass it.
- **No lockout timer** between attempts — the system exits after 3 failed attempts but can be restarted.
- Tested on Windows with Python 3.13. Linux/macOS compatibility not verified.

---

## Development Notes

> **Transparency note:** The concept, workflow, verification logic, UX decisions, and project direction are entirely my own. The implementation was coded with the assistance of Claude (Anthropic's AI), which I used as a coding tool — similar to how a developer might use an IDE plugin or pair programmer. Debugging, testing, threshold tuning, and iterative improvement were done through my own testing and judgment.

This is a learning and demonstration project, not a production security system.

---

## Possible Future Improvements

- [ ] Replace spectral fingerprinting with SpeechBrain speaker embeddings for more robust speaker verification
- [ ] Add lockout timer after failed attempts
- [ ] Support multiple enrolled users
- [ ] Add liveness detection using 3D depth estimation
- [ ] GUI interface instead of terminal

---

## Requirements

- Python 3.10+
- Webcam
- Microphone
- Windows (tested), Linux/macOS (untested)

---

## License

MIT License — free to use, modify, and distribute with attribution.