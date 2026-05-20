import os
import sys
import time
import json
import threading
import datetime
import pyaudio
import cv2
import numpy as np
import pyttsx3
from deepface import DeepFace
from vosk import Model, KaldiRecognizer

# Suppress TensorFlow/DeepFace startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ============================================================
#  CONFIG
# ============================================================
KNOWN_FACE_PATH    = os.path.join(os.path.dirname(__file__), "known_faces", "rimon1.jpg")
VOICE_PASSWORD     = "hello"
VOSK_MODEL_PATH    = "vosk-model-small-en-us-0.15"
VOICE_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "voice_profile.npy")
SCAN_TIMEOUT       = 20
MAX_ATTEMPTS       = 3
LOG_FILE           = os.path.join(os.path.dirname(__file__), "access_log.txt")
MAX_LOG_RECORDS    = 100
EYE_CLOSED_FRAMES_NEEDED = 2
VOICE_MATCH_THRESHOLD    = 0.82   # will auto-adjust based on enrollment
# ============================================================


# ──────────────────────────────────────────────
#  TERMINAL COLOURS
# ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def cprint(msg, colour=RESET):
    print(f"{colour}{msg}{RESET}")


# ──────────────────────────────────────────────
#  TEXT TO SPEECH
# ──────────────────────────────────────────────
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 190)
        voices = engine.getProperty('voices')
        female_voice = None
        for v in voices:
            if 'zira' in v.name.lower() or 'female' in v.name.lower() or 'hazel' in v.name.lower():
                female_voice = v.id
                break
        if female_voice is None and len(voices) > 1:
            female_voice = voices[1].id
        if female_voice:
            engine.setProperty('voice', female_voice)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        cprint(f"[TTS error]: {e}", RED)


# ──────────────────────────────────────────────
#  LOGGING
# ──────────────────────────────────────────────
def write_log(event):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_line  = f"[{timestamp}] {event}\n"
    existing  = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            existing = f.readlines()
    existing.append(new_line)
    if len(existing) > MAX_LOG_RECORDS:
        existing = existing[-MAX_LOG_RECORDS:]
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(existing)


# ──────────────────────────────────────────────
#  VOICE FEATURE EXTRACTION
# ──────────────────────────────────────────────
def extract_features(audio_data):
    audio = audio_data.astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    frame_size = 512
    hop_size   = 256
    features   = []

    for start in range(0, len(audio) - frame_size, hop_size):
        frame         = audio[start:start + frame_size]
        windowed      = frame * np.hanning(frame_size)
        spectrum      = np.abs(np.fft.rfft(windowed))
        bands         = np.array_split(spectrum, 20)
        band_energies = [np.mean(b**2) for b in bands]
        features.append(band_energies)

    if not features:
        return np.zeros(40)

    features = np.array(features)
    return np.concatenate([features.mean(axis=0), features.std(axis=0)])

def cosine_similarity(a, b):
    dot   = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(dot / denom)


# ──────────────────────────────────────────────
#  VOSK MODEL  (loaded once)
# ──────────────────────────────────────────────
_vosk_model = None

def load_vosk_model():
    global _vosk_model
    if _vosk_model is not None:
        return _vosk_model
    if not os.path.exists(VOSK_MODEL_PATH):
        cprint(f"[ERROR] Vosk model not found at '{VOSK_MODEL_PATH}'", RED)
        return None
    try:
        _vosk_model = Model(VOSK_MODEL_PATH)
        cprint("[OK] Voice model loaded.", GREEN)
    except Exception as e:
        cprint(f"[ERROR] Failed to load Vosk model: {e}", RED)
        return None
    return _vosk_model


# ──────────────────────────────────────────────
#  KNOWN FACE CHECK
# ──────────────────────────────────────────────
def check_known_face(path):
    if not os.path.exists(path):
        cprint(f"[ERROR] Known face image not found: {path}", RED)
        return False
    img = cv2.imread(path)
    if img is None:
        cprint(f"[ERROR] Could not read image: {path}", RED)
        return False
    cprint("[OK] Known face image found.", GREEN)
    return True


# ──────────────────────────────────────────────
#  VOICE PROFILE
# ──────────────────────────────────────────────
def load_voice_profile():
    if not os.path.exists(VOICE_PROFILE_PATH):
        cprint("[ERROR] Voice profile not found. Run enroll_voice.py first.", RED)
        return None
    profile = np.load(VOICE_PROFILE_PATH)
    cprint("[OK] Voice profile loaded.", GREEN)
    return profile


# ──────────────────────────────────────────────
#  STEP 1 — Camera permission
# ──────────────────────────────────────────────
def ask_camera_permission():
    cprint("\n" + "="*50, CYAN)
    cprint("   ZEPTO  —  3-Step Security System", BOLD)
    cprint("="*50 + "\n", CYAN)
    cprint("Asking for permission to turn on the camera.", YELLOW)
    speak("Asking for permission to turn on the camera.")
    while True:
        ans = input("Allow camera access? (yes / no): ").strip().lower()
        if ans in ("yes", "y"):
            cprint("\nThank you, now say the password.", GREEN)
            speak("Thank you. Look at the camera, say the password, and blink naturally.")
            return True
        elif ans in ("no", "n"):
            cprint("\nOk, cancelling process.", RED)
            speak("Ok, cancelling process.")
            write_log("CANCELLED — user denied camera access")
            return False
        else:
            cprint("Please type  yes  or  no.", YELLOW)


# ──────────────────────────────────────────────
#  VOICE LISTENER  (background thread)
#  FIX: raw_audio is stored per-chunk so main loop
#       can access it immediately after password detected
# ──────────────────────────────────────────────
class VoiceListener(threading.Thread):
    def __init__(self, model, password, timeout):
        super().__init__(daemon=True)
        self.model        = model
        self.password     = password.lower()
        self.timeout      = timeout
        self.password_ok  = False
        self.error        = False
        self._lock        = threading.Lock()
        self._frames      = []          # grows in real-time
        self._stop_event  = threading.Event()

    def stop(self):
        self._stop_event.set()

    def get_audio_so_far(self):
        """Return all captured audio as numpy array (thread-safe)."""
        with self._lock:
            if not self._frames:
                return np.array([], dtype=np.int16)
            return np.frombuffer(b"".join(self._frames), dtype=np.int16)

    def run(self):
        try:
            rec    = KaldiRecognizer(self.model, 16000)
            p      = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=16000,
                            input=True,
                            frames_per_buffer=4000)
            stream.start_stream()
            start = time.time()

            while not self._stop_event.is_set():
                if time.time() - start > self.timeout:
                    break
                data = stream.read(4000, exception_on_overflow=False)

                with self._lock:
                    self._frames.append(data)

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    heard  = result.get("text", "").strip().lower()
                    if heard:
                        cprint(f"\n[Voice] Heard: '{heard}'", CYAN)
                    if self.password in heard.split():
                        self.password_ok = True

            stream.stop_stream()
            stream.close()
            p.terminate()

        except Exception as e:
            cprint(f"\n[Voice ERROR] {e}", RED)
            self.error = True


# ──────────────────────────────────────────────
#  FACE MATCHER  (background thread)
# ──────────────────────────────────────────────
class FaceMatcher(threading.Thread):
    def __init__(self, frame):
        super().__init__(daemon=True)
        self.frame  = frame.copy()
        self.result = False
        self.done   = False

    def run(self):
        temp_path = f"temp_frame_{threading.get_ident()}.jpg"
        try:
            cv2.imwrite(temp_path, self.frame)
            r = DeepFace.verify(
                img1_path=temp_path,
                img2_path=KNOWN_FACE_PATH,
                model_name="VGG-Face",
                detector_backend="opencv",
                enforce_detection=False,
                silent=True
            )
            self.result = r.get("verified", False)
        except Exception:
            self.result = False
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            self.done = True


# ──────────────────────────────────────────────
#  COMBINED SCAN
# ──────────────────────────────────────────────
def run_combined_scan(vosk_model, voice_profile):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cprint("[ERROR] Cannot open webcam.", RED)
        return False

    for _ in range(5):
        cap.read()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    listener = VoiceListener(vosk_model, VOICE_PASSWORD, SCAN_TIMEOUT)
    listener.start()

    cprint("\n" + "-"*50, CYAN)
    cprint("  Scanning — say password & blink naturally", BOLD)
    cprint("-"*50, CYAN)
    cprint(f"  Say '{VOICE_PASSWORD}' out loud  |  Blink naturally  |  {SCAN_TIMEOUT}s\n", YELLOW)

    face_ok            = False
    blink_ok           = False
    speaker_ok         = False
    speaker_checked    = False
    eyes_closed_frames = 0
    eyes_were_closed   = False
    frame_count        = 0
    CHECK_EVERY        = 15
    face_thread        = None
    start_time         = time.time()

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > SCAN_TIMEOUT:
                cprint("\n[TIMEOUT] Time limit reached.", RED)
                break

            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── blink detection ──
            faces         = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            eye_detected  = len(faces) > 0
            eyes_open_now = True

            for (fx, fy, fw, fh) in faces:
                roi_gray = gray[fy:fy+fh, fx:fx+fw]
                eyes     = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
                if len(eyes) == 0:
                    eyes_closed_frames += 1
                    eyes_open_now = False
                else:
                    eyes_closed_frames = 0
                box_colour = (0, 255, 0) if face_ok else (0, 200, 255)
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), box_colour, 2)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (fx+ex, fy+ey), (fx+ex+ew, fy+ey+eh), (0, 255, 100), 1)

            if eyes_closed_frames >= EYE_CLOSED_FRAMES_NEEDED:
                eyes_were_closed = True
            if eyes_were_closed and eyes_open_now and eye_detected and not blink_ok:
                blink_ok           = True
                eyes_were_closed   = False
                eyes_closed_frames = 0
                cprint("\n[Liveness] Blink detected!", GREEN)

            # ── face recognition ──
            if not face_ok and frame_count % CHECK_EVERY == 0:
                if face_thread is None or face_thread.done:
                    face_thread = FaceMatcher(frame)
                    face_thread.start()

            if face_thread and face_thread.done and not face_ok:
                if face_thread.result:
                    face_ok = True
                    cprint("\n[Face] VERIFIED!", GREEN)

            # ── speaker verification ──
            # FIX: wait 1 second after password detected so enough audio is captured
            if listener.password_ok and not speaker_checked:
                audio = listener.get_audio_so_far()
                if len(audio) >= 16000:   # at least 1 second of audio
                    speaker_checked = True
                    features   = extract_features(audio)
                    similarity = cosine_similarity(features, voice_profile)
                    cprint(f"\n[Speaker] Similarity: {similarity:.3f} (threshold: {VOICE_MATCH_THRESHOLD})", CYAN)
                    if similarity >= VOICE_MATCH_THRESHOLD:
                        speaker_ok = True
                        cprint("[Speaker] VERIFIED!", GREEN)
                    else:
                        cprint("[Speaker] Voice did not match.", RED)

            # ── overlay ──
            remaining = int(SCAN_TIMEOUT - elapsed)

            if listener.password_ok and speaker_ok:
                voice_label = "Voice : VERIFIED  "
                vc = (0, 255, 0)
            elif speaker_checked and not speaker_ok:
                voice_label = "Voice : Wrong speaker"
                vc = (0, 0, 255)
            else:
                voice_label = "Voice : Listening..."
                vc = (0, 200, 255)

            face_label  = "Face  : VERIFIED  " if face_ok  else "Face  : Scanning..."
            blink_label = "Blink : VERIFIED  " if blink_ok else "Blink : Waiting..."
            time_label  = f"Time  : {remaining}s"
            fc = (0, 255, 0) if face_ok  else (0, 200, 255)
            bc = (0, 255, 0) if blink_ok else (0, 200, 255)

            cv2.rectangle(frame, (0, 0), (330, 115), (20, 20, 20), -1)
            cv2.putText(frame, face_label,  (10, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.62, fc, 2)
            cv2.putText(frame, blink_label, (10, 55),  cv2.FONT_HERSHEY_SIMPLEX, 0.62, bc, 2)
            cv2.putText(frame, voice_label, (10, 82),  cv2.FONT_HERSHEY_SIMPLEX, 0.62, vc, 2)
            cv2.putText(frame, time_label,  (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

            cv2.imshow("ZEPTO Security", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cprint("\n[INFO] Scan cancelled by user.", YELLOW)
                break

            if face_ok and blink_ok and listener.password_ok and speaker_ok:
                break

    finally:
        listener.stop()
        cap.release()
        cv2.destroyAllWindows()

    return face_ok and blink_ok and listener.password_ok and speaker_ok


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main():
    cprint("\n[INIT] Loading security modules...", CYAN)

    vosk_model = load_vosk_model()
    if vosk_model is None:
        sys.exit(1)

    if not check_known_face(KNOWN_FACE_PATH):
        sys.exit(1)

    voice_profile = load_voice_profile()
    if voice_profile is None:
        sys.exit(1)

    if not ask_camera_permission():
        sys.exit(0)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        cprint(f"\n[Attempt {attempt}/{MAX_ATTEMPTS}]", YELLOW)
        success = run_combined_scan(vosk_model, voice_profile)

        if success:
            cprint("\n" + "="*50, GREEN)
            cprint("  Access granted, Welcome Sir.", BOLD)
            cprint("="*50 + "\n", GREEN)
            speak("Access granted, Welcome Sir.")
            write_log("GRANTED — face, blink, voice and speaker verified")
            sys.exit(0)
        else:
            remaining = MAX_ATTEMPTS - attempt
            if remaining > 0:
                cprint(f"\nAccess denied. {remaining} attempt(s) left.", RED)
                speak("Access denied.")
                write_log(f"DENIED — attempt {attempt} failed")
                time.sleep(1)
            else:
                cprint("\n" + "="*50, RED)
                cprint("  Access denied.", BOLD)
                cprint("="*50 + "\n", RED)
                speak("Access denied.")
                write_log(f"DENIED — all {MAX_ATTEMPTS} attempts used")
                sys.exit(1)


if __name__ == "__main__":
    main()