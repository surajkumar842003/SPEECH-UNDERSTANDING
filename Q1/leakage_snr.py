import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate Test Signal (IMPORTANT)
# -----------------------------
# Single sinusoid → required for proper leakage analysis
N = 2048
t = np.arange(N)

# Frequency chosen NOT aligned with FFT bins → shows leakage clearly
freq = 50.5
signal = np.sin(2 * np.pi * freq * t / N)

# -----------------------------
# 2. Window Functions
# -----------------------------
windows = {
    "Rectangular": np.ones(N),
    "Hamming": np.hamming(N),
    "Hanning": np.hanning(N)
}

# -----------------------------
# 3. Spectral Leakage (CORRECT)
# -----------------------------
def spectral_leakage(signal, window):
    win_signal = signal * window
    fft = np.fft.fft(win_signal)
    mag = np.abs(fft)

    peak_idx = np.argmax(mag)

    # Exclude main lobe region (around peak)
    mask = np.ones_like(mag, dtype=bool)
    mask[max(0, peak_idx-2):peak_idx+3] = False

    leakage = np.sum(mag[mask]) / (np.sum(mag) + 1e-10)
    return leakage

# -----------------------------
# 4. SNR (CORRECT)
# -----------------------------
def compute_snr(clean_signal):
    noise = np.random.normal(0, 0.01, len(clean_signal))
    noisy_signal = clean_signal + noise

    snr = 10 * np.log10(
        np.sum(clean_signal**2) / (np.sum(noise**2) + 1e-10)
    )
    return snr

# -----------------------------
# 5. Analysis
# -----------------------------
leakage_results = {}
snr_results = {}

for name, w in windows.items():
    leakage_results[name] = spectral_leakage(signal, w)
    snr_results[name] = compute_snr(signal)

# -----------------------------
# 6. Print Results
# -----------------------------
print("Spectral Leakage:")
for k, v in leakage_results.items():
    print(f"{k}: {v:.4f}")

print("\nSNR:")
for k, v in snr_results.items():
    print(f"{k}: {v:.2f} dB")

# -----------------------------
# 7. Plot Results
# -----------------------------
names = list(windows.keys())

plt.figure(figsize=(10, 4))

# Leakage Plot
plt.subplot(1, 2, 1)
plt.bar(names, list(leakage_results.values()))
plt.title("Spectral Leakage Comparison")
plt.ylabel("Leakage")

# SNR Plot
plt.subplot(1, 2, 2)
plt.bar(names, list(snr_results.values()))
plt.title("SNR Comparison")
plt.ylabel("dB")

plt.tight_layout()
plt.savefig("leakage_snr_comparison.png")
plt.close()

print("\nPlot saved: leakage_snr_comparison.png ✅")