import numpy as np
from scipy.fftpack import dct
import soundfile as sf
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Pre-emphasis
# -----------------------------
def pre_emphasis(x, alpha=0.97):
    return np.append(x[0], x[1:] - alpha * x[:-1])

# -----------------------------
# 2. Framing (with padding)
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
# 3. Windowing
# -----------------------------
def apply_window(frames, window_type='hamming'):
    if window_type == 'hamming':
        window = np.hamming(frames.shape[1])
    elif window_type == 'hann':
        window = np.hanning(frames.shape[1])
    else:
        window = np.ones(frames.shape[1])

    return frames * window

# -----------------------------
# 4. FFT + Power Spectrum
# -----------------------------
def power_spectrum(frames, NFFT):
    fft = np.fft.rfft(frames, NFFT)
    return (1.0 / NFFT) * (np.abs(fft) ** 2)

# -----------------------------
# 5. Mel Filterbank
# -----------------------------
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def mel_filterbank(sr, NFFT, nfilt=40):
    low_mel = 0
    high_mel = hz_to_mel(sr / 2)

    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((nfilt, NFFT // 2 + 1))

    for m in range(1, nfilt + 1):
        f_m_minus, f_m, f_m_plus = bins[m-1], bins[m], bins[m+1]

        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-10)

        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-10)

    return fbank

# -----------------------------
# 6. Cepstrum (for later tasks)
# -----------------------------
def compute_cepstrum(frames):
    spectrum = np.fft.fft(frames)
    log_spec = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.ifft(log_spec).real
    return cepstrum

# -----------------------------
# 7. MFCC Extraction
# -----------------------------
def compute_mfcc(x, sr, nfilt=40, num_ceps=13, NFFT=512):

    x = pre_emphasis(x)
    frames = framing(x, sr)
    frames = apply_window(frames)

    # Energy
    energy = np.sum(frames**2, axis=1)
    energy = np.where(energy == 0, 1e-10, energy)

    # Power spectrum
    ps = power_spectrum(frames, NFFT)

    # Mel filterbank
    fb = mel_filterbank(sr, NFFT, nfilt)
    mel_energy = np.dot(ps, fb.T)
    mel_energy = np.where(mel_energy == 0, 1e-10, mel_energy)

    # Log
    log_energy = np.log(mel_energy)

    # DCT
    mfcc = dct(log_energy, type=2, axis=1, norm='ortho')[:, :num_ceps]

    # Replace first coefficient with log energy (optional but standard)
    mfcc[:, 0] = np.log(energy)

    # Normalize (important for visualization)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-10)

    return mfcc

# -----------------------------
# MAIN: process dataset
# -----------------------------
data_dir = "/DATA/suraj/pr2/suraj/data"

files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]

for file in files:
    path = os.path.join(data_dir, file)
    print("Processing:", path)

    audio, sr = sf.read(path)

    # Convert stereo → mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Normalize audio safely
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    mfcc = compute_mfcc(audio, sr)

    print("MFCC shape:", mfcc.shape)

    # -----------------------------
    # Plot MFCC
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc.T, aspect='auto', origin='lower', cmap='jet')
    plt.title(f"MFCC - {file}")
    plt.xlabel("Frames")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(f"mfcc_{file}.png")
    plt.close()

print("All files processed successfully ✅")