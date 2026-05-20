import os
import time
import numpy as np
import pyaudio
import pyttsx3
from scipy.io import wavfile
from scipy.signal import resample
import tempfile
import wave

# ============================================================
VOICE_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "voice_profile.npy")
SAMPLE_RATE        = 16000
RECORD_SECONDS     = 2.5
NUM_SAMPLES        = 3
# ============================================================

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def cprint(msg, colour=RESET):
    print(f"{colour}{msg}{RESET}")

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
    except Exception:
        pass

def extract_features(audio_data):
    """
    Extract MFCC-like frequency features from raw audio.
    Returns a 1D feature vector.
    """
    # normalize
    audio = audio_data.astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # compute FFT-based spectral features over frames
    frame_size   = 512
    hop_size     = 256
    features     = []

    for start in range(0, len(audio) - frame_size, hop_size):
        frame    = audio[start:start + frame_size]
        windowed = frame * np.hanning(frame_size)
        spectrum = np.abs(np.fft.rfft(windowed))
        # split spectrum into 20 bands and take mean energy per band
        bands    = np.array_split(spectrum, 20)
        band_energies = [np.mean(b**2) for b in bands]
        features.append(band_energies)

    if not features:
        return np.zeros(20)

    features = np.array(features)  # shape: (frames, 20)
    # return mean + std across frames = 40-dim vector
    return np.concatenate([features.mean(axis=0), features.std(axis=0)])

def record_sample(sample_num):
    """Record one voice sample and return feature vector."""
    p      = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=1024)

    cprint(f"\n  [{sample_num}/{NUM_SAMPLES}] Say 'hello' now...", YELLOW)
    speak("Say hello now.")
    time.sleep(0.3)

    frames = []
    for _ in range(int(SAMPLE_RATE / 1024 * RECORD_SECONDS)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # convert to numpy
    audio_np = np.frombuffer(b"".join(frames), dtype=np.int16)
    features = extract_features(audio_np)
    cprint(f"  Sample {sample_num} recorded.", GREEN)
    return features

def main():
    cprint("\n" + "="*50, CYAN)
    cprint("   ZEPTO — Voice Enrollment", BOLD)
    cprint("="*50, CYAN)
    cprint("\nThis will record your voice 3 times.", YELLOW)
    cprint("Each time, say the word:  hello\n", YELLOW)
    speak("Starting voice enrollment. You will say hello three times.")

    input("Press Enter when ready...")

    all_features = []
    for i in range(1, NUM_SAMPLES + 1):
        feat = record_sample(i)
        all_features.append(feat)
        time.sleep(0.8)

    # save mean profile
    profile = np.mean(all_features, axis=0)
    np.save(VOICE_PROFILE_PATH, profile)

    cprint(f"\n[OK] Voice profile saved to: voice_profile.npy", GREEN)
    cprint("[DONE] Enrollment complete. You can now run zepto_v2.py\n", GREEN)
    speak("Voice enrollment complete. You can now use the security system.")

if __name__ == "__main__":
    main()