import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load audio
# -----------------------------
data_dir = "/DATA/suraj/pr2/suraj/data"
file = [f for f in os.listdir(data_dir) if f.endswith(".wav")][0]

path = os.path.join(data_dir, file)
audio, sr = sf.read(path)

# Convert to mono
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

# Normalize
audio = audio / (np.max(np.abs(audio)) + 1e-10)

# -----------------------------
# Framing
# -----------------------------
def framing(x, sr, frame_size=0.025, stride=0.01):
    frame_len = int(frame_size * sr)
    step = int(stride * sr)

    num_frames = int(np.ceil((len(x) - frame_len) / step)) + 1
    pad_len = (num_frames - 1) * step + frame_len
    pad_signal = np.append(x, np.zeros(pad_len - len(x)))

    frames = []
    for i in range(num_frames):
        start = i * step
        frames.append(pad_signal[start:start + frame_len])

    return np.array(frames)

# -----------------------------
# Cepstrum
# -----------------------------
def compute_cepstrum(frame):
    spectrum = np.fft.fft(frame)
    log_spec = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.ifft(log_spec).real
    return cepstrum

# -----------------------------
# Voiced / Unvoiced Detection
# -----------------------------
def is_voiced(frame, sr):
    # Energy check (remove silence)
    energy = np.sum(frame**2)
    if energy < 1e-4:
        return 0

    cep = compute_cepstrum(frame)

    # Quefrency regions
    low_q = cep[:50]         # vocal tract
    high_q = cep[50:200]     # pitch region

    low_energy = np.sum(np.abs(low_q))
    high_energy = np.sum(np.abs(high_q))

    # Robust ratio
    ratio = high_energy / (low_energy + 1e-10)

    # Strict threshold (FIXED)
    return 1 if ratio > 0.6 else 0

# -----------------------------
# Segment Audio
# -----------------------------
def segment_audio(frames, sr):
    labels = []
    for f in frames:
        labels.append(is_voiced(f, sr))
    return np.array(labels)

# -----------------------------
# Run pipeline
# -----------------------------
frames = framing(audio, sr)
labels = segment_audio(frames, sr)

print("Total frames:", len(labels))
print("Voiced frames:", np.sum(labels))
print("Unvoiced frames:", len(labels) - np.sum(labels))

# -----------------------------
# Visualization
# -----------------------------
time = np.linspace(0, len(audio)/sr, len(audio))

plt.figure(figsize=(12, 4))
plt.plot(time, audio, alpha=0.5, label="Audio")

frame_step = int(0.01 * sr)

for i, label in enumerate(labels):
    start = i * frame_step
    end = start + frame_step

    if end < len(audio):
        if label == 1:
            plt.axvspan(start/sr, end/sr, color='green', alpha=0.2)
        else:
            plt.axvspan(start/sr, end/sr, color='red', alpha=0.1)

plt.title("Voiced (green) vs Unvoiced (red)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
np.save("labels.npy", labels)
print("Saved labels.npy ✅")
plt.savefig("voiced_unvoiced.png")
plt.close()

print("Plot saved: voiced_unvoiced.png ✅")