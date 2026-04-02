"""
evaluation_scripts/fad_eval.py — FAD + DNSMOS proxy validation.
All config from config.yaml + .env via config_loader.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import sqrtm
from scipy.signal import welch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from config_loader import CFG

SR           = CFG["audio"]["sample_rate"]
MAX_DUR      = CFG["audio"]["max_duration_sec"]
MAX_PAIRS    = CFG["dataset"]["max_eval_pairs"]
EXAMPLES_DIR = CFG["output"]["examples_dir"]
OUTPUT_DIR   = CFG["output"]["output_dir"]
EVAL_DIR     = CFG["output"]["eval_dir"]

os.makedirs(EVAL_DIR, exist_ok=True)


def extract_mel_features(paths, n_mels=128, n_fft=1024, hop=256):
    mel_tr   = T.MelSpectrogram(sample_rate=SR, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    features = []
    for path in tqdm(paths, desc="  Mel features"):
        try:
            w, s = torchaudio.load(path)
            if s != SR:
                w = T.Resample(s, SR)(w)
            if w.shape[0] > 1:
                w = w.mean(0, keepdim=True)
            w    = w[:, : SR * MAX_DUR]
            feat = mel_tr(w).log1p().squeeze(0).mean(dim=-1).numpy()
            features.append(feat)
        except Exception as e:
            print(f"  [warn] {path}: {e}")
    return np.array(features)


def compute_fad(real: np.ndarray, gen: np.ndarray) -> float:
    if len(real) < 2 or len(gen) < 2:
        return 0.0
    eps  = 1e-6 * np.eye(real.shape[1])
    mu1, sig1 = real.mean(0), np.cov(real, rowvar=False) + eps
    mu2, sig2 = gen.mean(0),  np.cov(gen,  rowvar=False) + eps
    diff     = mu1 - mu2
    covmean, _ = sqrtm(sig1 @ sig2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig1 + sig2 - 2.0 * covmean))


def compute_dnsmos_proxy(paths):
    rows = []
    for path in tqdm(paths, desc="  DNSMOS proxy"):
        try:
            w, s = torchaudio.load(path)
            if s != SR:
                w = T.Resample(s, SR)(w)
            wav = w.mean(0).numpy().astype(np.float64)

            freqs, psd  = welch(wav, fs=SR, nperseg=512)
            psd_pos      = psd[psd > 0]
            spec_flat    = (np.exp(np.mean(np.log(psd_pos + 1e-10)))
                            / (psd_pos.mean() + 1e-10))
            hf_ratio     = psd[freqs > 6000].mean() / (psd.mean() + 1e-10)

            rms = np.sqrt(np.mean(wav**2))
            n_frames = len(wav) // 512
            active   = sum(
                1 for f in range(n_frames)
                if np.sqrt(np.mean(wav[f*512:(f+1)*512]**2)) > 0.01 * rms
            )
            sar   = active / max(n_frames, 1)
            score = float(np.clip(
                2.0*sar + 1.5*(1 - min(spec_flat*10, 1)) + 1.5*(1 - min(hf_ratio, 1)),
                0.0, 5.0
            ))
            rows.append({
                "file": os.path.basename(path),
                "SAR": round(sar, 4),
                "SpectralFlat": round(float(spec_flat), 6),
                "HF_Ratio": round(float(hf_ratio), 6),
                "DNSMOS_proxy": round(score, 4)
            })
        except Exception as e:
            print(f"  [warn] {path}: {e}")
    return pd.DataFrame(rows)


def compute_snr(orig_paths, trans_paths):
    vals = []
    for op, tp in tqdm(zip(orig_paths, trans_paths),
                        total=len(orig_paths), desc="  SNR"):
        try:
            wo, so = torchaudio.load(op)
            wt, st = torchaudio.load(tp)
            if so != SR: wo = T.Resample(so, SR)(wo)
            if st != SR: wt = T.Resample(st, SR)(wt)
            ml = min(wo.shape[-1], wt.shape[-1])
            o  = wo.mean(0).numpy()[:ml].astype(np.float64)
            t  = wt.mean(0).numpy()[:ml].astype(np.float64)
            n  = o - t
            sp = np.mean(o**2); np_ = np.mean(n**2)
            vals.append(10*np.log10(sp/(np_+1e-12)) if np_ > 1e-12 and sp > 1e-12 else 40.0)
        except Exception as e:
            print(f"  [warn] {e}")
    return vals


def run_evaluation():
    all_files   = sorted(os.listdir(EXAMPLES_DIR)) if os.path.exists(EXAMPLES_DIR) else []
    orig_paths  = sorted([os.path.join(EXAMPLES_DIR, f) for f in all_files
                          if "_orig_" in f and f.endswith(".wav")])[:MAX_PAIRS]
    trans_paths = sorted([os.path.join(EXAMPLES_DIR, f) for f in all_files
                          if "_trans_" in f and f.endswith(".wav")])[:MAX_PAIRS]

    n = min(len(orig_paths), len(trans_paths))
    if n == 0:
        print("[fad_eval] ⚠ No audio pairs found. Run pp_demo.py first.")
        return

    print(f"[fad_eval] Evaluating {n} audio pairs ...")
    orig_paths  = orig_paths[:n]
    trans_paths = trans_paths[:n]

    print("\n[fad_eval] FAD ...")
    real_f    = extract_mel_features(orig_paths)
    gen_f     = extract_mel_features(trans_paths)
    fad_score = compute_fad(real_f, gen_f)

    verdict = ("EXCELLENT" if fad_score < 20 else
               "GOOD"      if fad_score < 100 else
               "MODERATE"  if fad_score < 300 else "POOR")
    print(f"  FAD = {fad_score:.4f}  [{verdict}]")

    print("\n[fad_eval] DNSMOS proxy ...")
    dn_orig  = compute_dnsmos_proxy(orig_paths)
    dn_trans = compute_dnsmos_proxy(trans_paths)
    mo, mt   = dn_orig["DNSMOS_proxy"].mean(), dn_trans["DNSMOS_proxy"].mean()
    drop     = mo - mt
    print(f"  DNSMOS orig={mo:.4f}  trans={mt:.4f}  drop={drop:.4f}  "
          f"[{'PASS' if drop < 0.3 else 'FAIL'}]")

    print("\n[fad_eval] SNR ...")
    snrs     = compute_snr(orig_paths, trans_paths)
    mean_snr = float(np.mean(snrs)) if snrs else 0.0
    print(f"  Mean SNR = {mean_snr:.2f} dB")

    # Save
    summary = pd.DataFrame({
        "Metric":  ["FAD", "DNSMOS_orig", "DNSMOS_trans", "DNSMOS_drop", "SNR_dB"],
        "Value":   [round(fad_score,4), round(mo,4), round(mt,4), round(drop,4), round(mean_snr,2)],
        "Status":  [verdict,
                    "Good" if mo>3 else "Moderate",
                    "Good" if mt>3 else "Moderate",
                    "PASS" if drop<0.3 else "FAIL",
                    "PASS" if mean_snr>5 else "CHECK"]
    })
    summary.to_csv(os.path.join(EVAL_DIR, "validation_results.csv"), index=False)
    dn_orig.to_csv(os.path.join(EVAL_DIR,  "dnsmos_original.csv"),    index=False)
    dn_trans.to_csv(os.path.join(EVAL_DIR, "dnsmos_transformed.csv"), index=False)
    print("\n[fad_eval] Validation Summary:")
    print(summary.to_string(index=False))

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Part 4: FAD & DNSMOS Validation", fontsize=14, fontweight="bold")

    fc = "#2ca02c" if fad_score<100 else ("#ff7f0e" if fad_score<300 else "#d62728")
    axes[0,0].bar(["FAD Score"], [fad_score], color=fc, width=0.4, edgecolor="black")
    axes[0,0].axhline(100, color="orange", ls="--", label="Good (100)")
    axes[0,0].axhline(300, color="red",    ls="--", label="Poor (300)")
    axes[0,0].set_title("Frechet Audio Distance (↓ better)")
    axes[0,0].text(0, fad_score*1.02, f"{fad_score:.2f}", ha="center", fontweight="bold")
    axes[0,0].legend()

    axes[0,1].bar(["Original","Transformed"], [mo, mt],
                  color=["#4C72B0","#DD8452"], edgecolor="black")
    axes[0,1].set_ylim(0, 5.5); axes[0,1].set_title("DNSMOS Proxy (↑ better, max 5)")
    axes[0,1].axhline(3.0, color="green", ls="--", alpha=0.6, label="Acceptable (3)")
    axes[0,1].legend()
    for i,v in enumerate([mo, mt]):
        axes[0,1].text(i, v+0.05, f"{v:.3f}", ha="center", fontweight="bold")

    if snrs:
        axes[1,0].hist(snrs, bins=15, color="#4C72B0", edgecolor="black", alpha=0.8)
        axes[1,0].axvline(mean_snr, color="red", ls="--", label=f"Mean={mean_snr:.1f}dB")
        axes[1,0].set_title("SNR Distribution"); axes[1,0].set_xlabel("dB")
        axes[1,0].legend()

    axes[1,1].plot(real_f.mean(0), color="blue",   lw=1.5, label="Original")
    axes[1,1].plot(gen_f.mean(0),  color="orange", lw=1.5, ls="--", label="Transformed")
    axes[1,1].fill_between(range(real_f.shape[1]),
        real_f.mean(0)-real_f.std(0), real_f.mean(0)+real_f.std(0), alpha=0.15, color="blue")
    axes[1,1].fill_between(range(gen_f.shape[1]),
        gen_f.mean(0)-gen_f.std(0), gen_f.mean(0)+gen_f.std(0), alpha=0.15, color="orange")
    axes[1,1].set_title("Mel Feature Distribution"); axes[1,1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "fad_dnsmos_validation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fad_eval] Saved: {EVAL_DIR}/fad_dnsmos_validation.png")
    return summary


if __name__ == "__main__":
    run_evaluation()
    print("[fad_eval] ✓ Part 4 complete.")
