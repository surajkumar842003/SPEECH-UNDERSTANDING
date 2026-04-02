"""
audit.py — Part 1: LibriSpeech Documentation Debt & Representation Bias Audit
Reads all parameters from config.yaml + .env via config_loader.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import seaborn as sns
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from config_loader import CFG

# ── Config ────────────────────────────────────────────────────
DATA_ROOT          = CFG["dataset"]["data_root"]
DATASET_URL        = CFG["dataset"]["url"]
MAX_AUDIT_SAMPLES  = CFG["dataset"]["max_audit_samples"]
OUTPUT_DIR         = CFG["output"]["output_dir"]
SEED               = CFG["seed"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_ROOT,  exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)


def download_dataset():
    print(f"[audit] Downloading LibriSpeech '{DATASET_URL}' ...")
    dataset = torchaudio.datasets.LIBRISPEECH(DATA_ROOT, url=DATASET_URL, download=True)
    print(f"[audit] Total samples: {len(dataset)}")
    return dataset


def parse_speakers_txt():
    path = os.path.join(DATA_ROOT, "LibriSpeech", "SPEAKERS.TXT")
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(";") or not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    minutes = float(parts[3])
                except ValueError:
                    minutes = 0.0
                rows.append({
                    "speaker_id": int(parts[0]),
                    "gender":     parts[1].strip().upper(),
                    "subset":     parts[2].strip(),
                    "minutes":    minutes,
                    "name":       parts[4].strip() if len(parts) > 4 else ""
                })
    return pd.DataFrame(rows)


def run_audit(dataset, speakers_df):
    subset_key = DATASET_URL  # e.g. "test-clean"
    train_df   = speakers_df[speakers_df["subset"] == subset_key].copy()

    # Fallback: if subset not found in SPEAKERS.TXT (test-clean IDs are listed)
    # collect speaker IDs from actual dataset
    if len(train_df) == 0:
        print(f"[audit] Subset '{subset_key}' not found in SPEAKERS.TXT directly. "
              f"Extracting speaker IDs from dataset ...")
        spk_ids_in_dataset = set()
        n_check = min(len(dataset), MAX_AUDIT_SAMPLES)
        for i in tqdm(range(n_check), desc="  Collecting speaker IDs"):
            _, _, _, spk_id, _, _ = dataset[i]
            spk_ids_in_dataset.add(spk_id)
        train_df = speakers_df[speakers_df["speaker_id"].isin(spk_ids_in_dataset)].copy()
        print(f"[audit] Found {len(train_df)} speakers via ID matching.")

    print(f"\n[audit] Speakers in '{subset_key}': {len(train_df)}")

    gender_counts  = train_df["gender"].value_counts()
    gender_pct     = (gender_counts / gender_counts.sum() * 100).round(2)
    time_by_gender = (
        train_df.groupby("gender")["minutes"]
        .agg(Total="sum", Mean="mean", Count="count")
        .round(2)
    )

    male_pct   = gender_pct.get("M", 0.0)
    female_pct = gender_pct.get("F", 0.0)
    imbalance  = abs(male_pct - female_pct)

    print(f"\n[audit] === Documentation Debt Report ===")
    print(f"  Speakers (M/F): {gender_counts.get('M',0)}/{gender_counts.get('F',0)}")
    print(f"  % Male: {male_pct:.1f}%  |  % Female: {female_pct:.1f}%")
    print(f"  Gender imbalance gap: {imbalance:.1f}%")
    print(f"  Age labels: MISSING (100% debt)")
    print(f"  Dialect labels: MISSING (100% debt)")
    print(f"  Accent labels: MISSING (100% debt)")
    if imbalance > 10:
        print("  ⚠ WARNING: Significant gender imbalance detected!")
    else:
        print("  ✓ Gender distribution relatively balanced.")

    # ── Audio duration stats ──────────────────────────────────
    n_sample  = min(MAX_AUDIT_SAMPLES, len(dataset))
    indices   = random.sample(range(len(dataset)), n_sample)
    durations = []
    print(f"\n[audit] Sampling {n_sample}/{len(dataset)} utterances for audio stats ...")
    for idx in tqdm(indices, desc="  Reading audio"):
        waveform, sr, _, _, _, _ = dataset[idx]
        durations.append(waveform.shape[-1] / sr)
    dur_arr = np.array(durations)

    print(f"[audit] Duration stats (N={n_sample}): "
          f"mean={dur_arr.mean():.2f}s  std={dur_arr.std():.2f}s  "
          f"min={dur_arr.min():.2f}s  max={dur_arr.max():.2f}s")

    # ── Save CSVs ─────────────────────────────────────────────
    train_df.to_csv(os.path.join(OUTPUT_DIR, "speakers_audit.csv"), index=False)
    time_by_gender.to_csv(os.path.join(OUTPUT_DIR, "gender_time_audit.csv"))

    debt_df = pd.DataFrame({
        "Metadata Field":  ["Gender", "Age", "Dialect", "Accent", "Noise Level", "Recording Env"],
        "Available":       ["Yes", "No", "No", "No", "No", "No"],
        "Coverage %":      [100, 0, 0, 0, 0, 0],
        "Quality (0-10)":  [7, 0, 0, 0, 0, 0],
        "Debt Severity":   ["Low", "High", "High", "High", "High", "High"]
    })
    debt_df.to_csv(os.path.join(OUTPUT_DIR, "documentation_debt.csv"), index=False)
    print(f"[audit] CSVs saved.")

    # ── Plots ─────────────────────────────────────────────────
    colors = ["#4C72B0", "#DD8452"]

    def _make_fig():
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"LibriSpeech '{subset_key}' — Bias & Documentation Debt Audit\n"
            f"(N={n_sample} utterances, {len(train_df)} speakers)",
            fontsize=14, fontweight="bold"
        )

        # 1: speaker count
        ax = axes[0, 0]
        bars = ax.bar(gender_counts.index, gender_counts.values,
                      color=colors, edgecolor="black", width=0.5)
        ax.set_title("Speaker Count by Gender"); ax.set_xlabel("Gender"); ax.set_ylabel("Speakers")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(int(bar.get_height())), ha="center", fontweight="bold")

        # 2: pie
        ax = axes[0, 1]
        ax.pie(gender_counts.values, labels=gender_counts.index,
               autopct="%1.1f%%", colors=colors, startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title("Gender Distribution (%)")

        # 3: speaking time
        ax = axes[0, 2]
        ax.bar(time_by_gender.index, time_by_gender["Total"],
               color=colors, edgecolor="black", width=0.5)
        ax.set_title("Total Speaking Time by Gender (min)")
        ax.set_xlabel("Gender"); ax.set_ylabel("Total Minutes")

        # 4: duration histogram
        ax = axes[1, 0]
        ax.hist(durations, bins=25, color="#2ca02c", edgecolor="black", alpha=0.8)
        ax.axvline(dur_arr.mean(), color="red", linestyle="--",
                   label=f"Mean={dur_arr.mean():.2f}s")
        ax.set_title(f"Utterance Duration Distribution (N={n_sample})")
        ax.set_xlabel("Duration (s)"); ax.set_ylabel("Count")
        ax.legend()

        # 5: debt heatmap
        ax = axes[1, 1]
        heat_df = debt_df.set_index("Metadata Field")[["Coverage %", "Quality (0-10)"]]
        sns.heatmap(heat_df, annot=True, fmt=".0f", cmap="RdYlGn",
                    vmin=0, vmax=10, ax=ax, linewidths=0.5)
        ax.set_title("Documentation Debt Matrix")

        # 6: balance vs ideal
        ax = axes[1, 2]
        ax.bar(["Male %", "Female %"], [male_pct, female_pct],
               color=colors, edgecolor="black", width=0.5)
        ax.axhline(50, color="gray", linestyle="--", alpha=0.7, label="Ideal 50%")
        ax.set_title("Gender Balance vs. Ideal 50/50")
        ax.set_ylabel("Percentage (%)"); ax.set_ylim(0, 100); ax.legend()
        for i, v in enumerate([male_pct, female_pct]):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

        plt.tight_layout()
        return fig

    fig1 = _make_fig()
    png_path = os.path.join(OUTPUT_DIR, "audit_plots.png")
    fig1.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    pdf_path = os.path.join(OUTPUT_DIR, "audit_plots.pdf")
    with pdf_backend.PdfPages(pdf_path) as pp:
        fig2 = _make_fig()
        pp.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    print(f"[audit] Saved: {png_path}")
    print(f"[audit] Saved: {pdf_path}")

    return train_df, indices, durations


def main():
    dataset   = download_dataset()
    spk_df    = parse_speakers_txt()
    train_df, sample_indices, durations = run_audit(dataset, spk_df)
    print("[audit] Audit complete.")
    return dataset, train_df, sample_indices


if __name__ == "__main__":
    main()
