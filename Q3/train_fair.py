"""
train_fair.py — Fairness-Aware ASR Training with Custom FairnessLoss.
All hyperparameters from config.yaml + .env.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer
import warnings
warnings.filterwarnings("ignore")

from config_loader import CFG

DATA_ROOT       = CFG["dataset"]["data_root"]
DATASET_URL     = CFG["dataset"]["url"]
N_TRAIN         = CFG["dataset"]["max_train_samples"]
SR              = CFG["audio"]["sample_rate"]
MAX_DUR         = CFG["audio"]["max_duration_sec"]
BATCH_SIZE      = CFG["training"]["batch_size"]
N_EPOCHS        = CFG["training"]["n_epochs"]
LR              = CFG["training"]["learning_rate"]
LAMBDA_FAIR     = CFG["training"]["lambda_fair"]
GRAD_CLIP       = CFG["training"]["grad_clip"]
MODEL_NAME      = CFG["training"]["model_name"]
FREEZE_FEAT     = CFG["training"]["freeze_feature_encoder"]
MODEL_SAVE_DIR  = CFG["training"]["model_save_dir"]
OUTPUT_DIR      = CFG["output"]["output_dir"]
SEED            = CFG["seed"]

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,     exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


class FairnessLoss(nn.Module):
    """
    Total Loss = mean(CTC) + λ × Var(per_group_mean_CTC)
    Minimises WER gap between demographic groups (gender).
    """
    def __init__(self, lambda_fair: float = 0.5, blank: int = 0):
        super().__init__()
        self.lambda_fair = lambda_fair
        self.ctc         = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="none")

    def forward(self, log_probs, targets, input_lengths, target_lengths, group_ids):
        B = log_probs.size(1)
        per_sample, offset = [], 0

        for b in range(B):
            t_len = target_lengths[b].item()
            if t_len == 0:
                per_sample.append(torch.tensor(0.0, device=log_probs.device, requires_grad=True))
                continue
            loss_b = self.ctc(
                log_probs[:, b:b+1, :],
                targets[offset: offset + t_len].unsqueeze(0),
                input_lengths[b:b+1],
                target_lengths[b:b+1]
            )
            per_sample.append(loss_b.squeeze())
            offset += t_len

        per_sample_t = torch.stack(per_sample)
        base_loss    = per_sample_t.mean()

        unique_groups = sorted(set(group_ids))
        group_means   = []
        for g in unique_groups:
            mask = torch.tensor(
                [1.0 if gid == g else 0.0 for gid in group_ids],
                device=log_probs.device
            )
            if mask.sum() > 0:
                group_means.append((per_sample_t * mask).sum() / mask.sum())

        fairness_penalty = (
            torch.stack(group_means).var() if len(group_means) >= 2
            else torch.tensor(0.0, device=log_probs.device)
        )

        return base_loss + self.lambda_fair * fairness_penalty, base_loss, fairness_penalty


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


def load_sample(dataset, idx, processor):
    waveform, sr, transcript, spk_id, _, _ = dataset[idx]
    if sr != SR:
        waveform = T.Resample(orig_freq=sr, new_freq=SR)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform[:, : SR * MAX_DUR]
    return waveform.squeeze().numpy(), transcript, spk_id


def compute_wer_by_group(dataset, indices, model, processor, spk_gender):
    model.eval()
    results = defaultdict(list)
    with torch.no_grad():
        for idx in tqdm(indices[:80], desc="  WER evaluation"):
            wav_np, transcript, spk_id = load_sample(dataset, idx, processor)
            inputs   = processor(wav_np, sampling_rate=SR,
                                 return_tensors="pt", padding=True).to(DEVICE)
            logits   = model(**inputs).logits
            pred_str = processor.batch_decode(logits.argmax(dim=-1))[0].lower()
            score    = min(wer(transcript.lower(), pred_str), 5.0)
            results[spk_gender.get(spk_id, "M")].append(score)
    model.train()
    return {g: {"mean_WER": round(float(np.mean(v)), 4), "count": len(v)}
            for g, v in results.items()}


def run_training():
    print(f"[train_fair] Device: {DEVICE}")
    print(f"[train_fair] Config: N_TRAIN={N_TRAIN}, BATCH={BATCH_SIZE}, "
          f"EPOCHS={N_EPOCHS}, λ={LAMBDA_FAIR}")

    dataset    = torchaudio.datasets.LIBRISPEECH(DATA_ROOT, url=DATASET_URL, download=True)
    spk_gender = get_speaker_gender_map()

    print(f"[train_fair] Loading {MODEL_NAME} ...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model     = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE)

    if FREEZE_FEAT:
        model.freeze_feature_encoder()

    model.train()
    optimizer    = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    fair_loss_fn = FairnessLoss(
        lambda_fair=LAMBDA_FAIR, blank=processor.tokenizer.pad_token_id
    )

    all_indices = random.sample(range(len(dataset)), min(N_TRAIN, len(dataset)))
    wer_before  = compute_wer_by_group(dataset, all_indices, model, processor, spk_gender)
    print(f"[train_fair] Baseline WER: {wer_before}")

    train_records, best_loss = [], float("inf")

    for epoch in range(N_EPOCHS):
        ep_indices = random.sample(all_indices, min(N_TRAIN, len(all_indices)))
        n_batches  = len(ep_indices) // BATCH_SIZE

        for b_idx in tqdm(range(n_batches), desc=f"  Epoch {epoch+1}/{N_EPOCHS}"):
            batch = ep_indices[b_idx * BATCH_SIZE: (b_idx + 1) * BATCH_SIZE]
            if len(batch) < 2:
                continue

            optimizer.zero_grad()
            wavs, transcripts, spk_ids = [], [], []
            for idx in batch:
                w, t, s = load_sample(dataset, idx, processor)
                wavs.append(w); transcripts.append(t); spk_ids.append(s)

            max_len    = max(w.shape[0] for w in wavs)
            wavs_pad   = [np.pad(w, (0, max_len - w.shape[0])) for w in wavs]

            inputs   = processor(wavs_pad, sampling_rate=SR,
                                 return_tensors="pt", padding=True).to(DEVICE)
            logits   = model(**inputs).logits
            log_prob = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

            with processor.as_target_processor():
                lbl_batch = processor(transcripts, return_tensors="pt", padding=True)
            label_ids = lbl_batch["input_ids"]

            flat_tgts, tgt_lens = [], []
            pad_id = processor.tokenizer.pad_token_id
            for row in label_ids:
                row = row[row != pad_id]
                flat_tgts.append(row); tgt_lens.append(len(row))

            targets_cat   = torch.cat(flat_tgts).to(DEVICE)
            tgt_lens_t    = torch.tensor(tgt_lens,  dtype=torch.long, device=DEVICE)
            T_len         = log_prob.size(0)
            inp_lens_t    = torch.full((len(batch),), T_len, dtype=torch.long, device=DEVICE)
            groups        = [spk_gender.get(s, "M") for s in spk_ids]

            loss, base, fp = fair_loss_fn(log_prob, targets_cat, inp_lens_t, tgt_lens_t, groups)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [warn] NaN/Inf at epoch {epoch+1} batch {b_idx} — skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            rec = {
                "epoch": epoch+1, "batch": b_idx+1,
                "total_loss": round(loss.item(), 5),
                "ctc_loss":   round(base.item(), 5),
                "fairness":   round(fp.item(), 7)
            }
            train_records.append(rec)

            if loss.item() < best_loss:
                best_loss = loss.item()
                model.save_pretrained(MODEL_SAVE_DIR)
                processor.save_pretrained(MODEL_SAVE_DIR)

    print(f"\n[train_fair] Best loss: {best_loss:.5f} → saved to {MODEL_SAVE_DIR}/")

    wer_after = compute_wer_by_group(dataset, all_indices, model, processor, spk_gender)
    print(f"[train_fair] Post-training WER: {wer_after}")

    # Save results
    train_df = pd.DataFrame(train_records)
    train_df.to_csv(os.path.join(OUTPUT_DIR, "training_losses.csv"), index=False)

    wer_rows = []
    for g in sorted(set(list(wer_before) + list(wer_after))):
        wer_rows.append({
            "gender":     g,
            "WER_before": wer_before.get(g, {}).get("mean_WER", None),
            "WER_after":  wer_after.get(g, {}).get("mean_WER", None),
            "count":      wer_before.get(g, {}).get("count", 0)
        })
    wer_df = pd.DataFrame(wer_rows)
    wer_df.to_csv(os.path.join(OUTPUT_DIR, "wer_by_gender.csv"), index=False)
    print("\n[train_fair] WER by Gender:")
    print(wer_df.to_string(index=False))

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Part 3: Fairness-Aware ASR Training\n"
                 f"Dataset: {DATASET_URL} | λ={LAMBDA_FAIR} | N={N_TRAIN}",
                 fontsize=13, fontweight="bold")

    axes[0].plot(train_df["ctc_loss"].values,  "b-o", ms=3, label="CTC Loss")
    axes[0].plot(train_df["total_loss"].values, "r-s", ms=3, label="Total Loss")
    axes[0].set_title("Loss Curves"); axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_df["fairness"].values, "g-^", ms=3)
    axes[1].set_title("Fairness Penalty"); axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Var(Group Losses)"); axes[1].grid(True, alpha=0.3)

    x = np.arange(len(wer_df)); w = 0.35
    axes[2].bar(x - w/2, wer_df["WER_before"].fillna(0), w, label="Before", color="#4C72B0")
    axes[2].bar(x + w/2, wer_df["WER_after"].fillna(0),  w, label="After",  color="#DD8452")
    axes[2].set_xticks(x); axes[2].set_xticklabels(wer_df["gender"])
    axes[2].set_title("WER Before vs After"); axes[2].set_ylabel("WER")
    axes[2].legend(); axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_results.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return model, processor, wer_df


if __name__ == "__main__":
    run_training()
    print("[train_fair] ✓ Part 3 complete.")
