"""
pp_demo.py — Privacy-Preserving Demo: processes audio pairs, saves WAVs,
generates spectrogram plots. All config from config.yaml + .env.
"""

import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from config_loader import CFG
from privacymodule import VoiceBiometricObfuscator, save_audio_wav

DATA_ROOT    = CFG["dataset"]["data_root"]
DATASET_URL  = CFG["dataset"]["url"]
N_PAIRS      = CFG["dataset"]["max_demo_pairs"]
SR           = CFG["audio"]["sample_rate"]
MAX_DUR      = CFG["audio"]["max_duration_sec"]
EXAMPLES_DIR = CFG["output"]["examples_dir"]
OUTPUT_DIR   = CFG["output"]["output_dir"]
SEED         = CFG["seed"]

os.makedirs(EXAMPLES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,   exist_ok=True)
random.seed(SEED)


def get_speaker_gender_map():
    path = os.path.join(DATA_ROOT, "LibriSpeech", "SPEAKERS.TXT")
    m = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(";") or not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[0].isdigit():
                m[int(parts[0])] = parts[1].strip().upper()
    return m


def run_demo():
    print(f"[pp_demo] Loading LibriSpeech '{DATASET_URL}' ...")
    dataset    = torchaudio.datasets.LIBRISPEECH(DATA_ROOT, url=DATASET_URL, download=True)
    spk_gender = get_speaker_gender_map()

    model = VoiceBiometricObfuscator()
    model.eval()

    n_use   = min(N_PAIRS, len(dataset))
    indices = random.sample(range(len(dataset)), n_use)
    records = []

    print(f"[pp_demo] Processing {n_use} audio pairs ...")
    for i, idx in enumerate(tqdm(indices, desc="  Transforming")):
        waveform, sr, transcript, spk_id, chapter_id, utt_id = dataset[idx]

        if sr != SR:
            waveform = T.Resample(orig_freq=sr, new_freq=SR)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform[:, : SR * MAX_DUR]

        gender    = spk_gender.get(spk_id, "M")
        direction = "male_to_female" if gender == "M" else "female_to_male"
        label     = "M2F" if gender == "M" else "F2M"

        orig_path  = os.path.join(EXAMPLES_DIR, f"pair_{i:03d}_orig_{gender}.wav")
        trans_path = os.path.join(EXAMPLES_DIR, f"pair_{i:03d}_trans_{label}.wav")

        save_audio_wav(waveform, orig_path)

        with torch.no_grad():
            transformed = model(waveform, direction=direction)
        save_audio_wav(transformed, trans_path)

        records.append({
            "pair_idx":   i,
            "speaker_id": spk_id,
            "gender":     gender,
            "direction":  label,
            "transcript": transcript[:80],
            "orig_path":  orig_path,
            "trans_path": trans_path,
            "dur_s":      round(waveform.shape[-1] / SR, 3),
        })

    meta_df = pd.DataFrame(records)
    meta_df.to_csv(os.path.join(OUTPUT_DIR, "transformation_metadata.csv"), index=False)
    print(f"[pp_demo] {len(records)} pairs saved to {EXAMPLES_DIR}/")

    # ── Spectrogram plots ─────────────────────────────────────
    mel_tr  = T.MelSpectrogram(sample_rate=SR, n_fft=512, hop_length=128, n_mels=80)
    n_plot  = min(4, len(records))
    fig, axes = plt.subplots(n_plot, 2, figsize=(14, 4 * n_plot))
    if n_plot == 1:
        axes = np.array([axes])
    fig.suptitle("Privacy-Preserving Transformation: Spectrogram Pairs",
                 fontsize=14, fontweight="bold")

    for row, rec in enumerate(records[:n_plot]):
        w_o, _ = torchaudio.load(rec["orig_path"])
        w_t, _ = torchaudio.load(rec["trans_path"])
        s_o = mel_tr(w_o).squeeze().log1p().numpy()
        s_t = mel_tr(w_t).squeeze().log1p().numpy()
        vmin = min(s_o.min(), s_t.min())
        vmax = max(s_o.max(), s_t.max())

        axes[row, 0].imshow(s_o, aspect="auto", origin="lower",
                            cmap="magma", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"Original — Spk {rec['speaker_id']} ({rec['gender']})", fontsize=9)
        axes[row, 0].set_xlabel("Time Frame"); axes[row, 0].set_ylabel("Mel Bin")

        axes[row, 1].imshow(s_t, aspect="auto", origin="lower",
                            cmap="magma", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"Transformed ({rec['direction']}) — Biometrics Obfuscated", fontsize=9)
        axes[row, 1].set_xlabel("Time Frame"); axes[row, 1].set_ylabel("Mel Bin")

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "spectrogram_pairs.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[pp_demo] Saved: {out_png}")
    return meta_df


if __name__ == "__main__":
    run_demo()
    print("[pp_demo] ✓ Part 2 complete.")
