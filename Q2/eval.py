
import argparse, os, sys, json, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import LibriSpeakerDataset
from utils   import load_config, set_seed, compute_eer, compute_mindcf
from torch.utils.data import DataLoader, Dataset
import torchaudio


def _loss(cfg, *keys, default=0.2):
    d = cfg.get("loss", {})
    for k in keys:
        if k in d: return d[k]
    return default

def _model_cfg(cfg, *keys, default=None):
    d = cfg.get("model", {})
    for k in keys:
        if k in d: return d[k]
    return default


class LibriSpeechTestDataset(Dataset):
    
    def __init__(self, cfg, root="data", url="test-clean"):
        self.cfg  = cfg
        self.root = root
        sr        = cfg["data"]["sample_rate"]
        n_mels    = cfg["data"]["n_mels"]
        n_fft     = cfg["data"]["n_fft"]
        win       = cfg["data"]["win_length"]
        hop       = cfg["data"]["hop_length"]
        max_f     = cfg["data"].get("max_frames", 200)
        self.max_samples = max_f * hop

        print(f"  [test-clean] root={root}  url={url}")
        print(f"  [test-clean] Downloading / verifying (334 MB) ...")
        self.ds = torchaudio.datasets.LIBRISPEECH(
            root=root, url=url, download=True)
        print(f"  [test-clean] Dataset ready. Samples={len(self.ds)}")

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft,
            win_length=win, hop_length=hop,
            n_mels=n_mels)
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

        spk_set = sorted(set(
            self.ds[i][3] for i in range(len(self.ds))))
        self.spk2idx = {s: i for i, s in enumerate(spk_set)}
        self.num_speakers = len(spk_set)
        print(f"  [test-clean] Speakers={self.num_speakers}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        waveform, sr, _, spk_id, _, _ = self.ds[idx]
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            pad = self.max_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad))
        mel = self.amp_to_db(self.mel_transform(waveform))  # (1, n_mels, T)
        mel = mel.squeeze(0)                                 # (n_mels, T)
        return mel, self.spk2idx[spk_id]


def sniff_arch(state_dict):
    keys = list(state_dict.keys())
    info = {}
    is_dis = any("ae." in k or "env_disc" in k or "spk_cls" in k
                 for k in keys)
    info["type"] = "disentangled" if is_dis else "baseline"

    if info["type"] == "baseline":
        for k in keys:
            w = state_dict[k]
            if w.dim() == 2 and ("classifier" in k or "spk_cls" in k) \
                    and "weight" in k:
                info["num_spk"] = w.shape[0]
                info["emb_dim"] = w.shape[1]
                break
        return info

    for k in keys:
        if "spk_cls" in k and k.endswith(".W"):
            w = state_dict[k]
            info["num_spk"] = w.shape[0]
            info["spk_dim"] = w.shape[1]
            break

    for k in keys:
        if "ae.enc_spk" in k and "weight" in k:
            w = state_dict[k]
            if w.dim() == 2:
                info["emb_dim"] = w.shape[1]
                break
    if "emb_dim" not in info:
        for k in keys:
            if "ae.decoder" in k and "weight" in k:
                w = state_dict[k]
                if w.dim() == 2:
                    info["emb_dim"] = w.shape[0]
                    break

    if "spk_dim" not in info:
        for k in sorted(keys):
            if "ae.enc_spk" in k and "weight" in k:
                w = state_dict[k]
                if w.dim() == 2:
                    info["spk_dim"] = w.shape[0]
                    break

    for k in sorted(keys):
        if "ae.enc_env" in k and "weight" in k:
            w = state_dict[k]
            if w.dim() == 2:
                info["env_dim"] = w.shape[0]
                break

    for k in keys:
        if "env_disc_ee" in k and "weight" in k:
            w = state_dict[k]
            if w.dim() == 2:
                info["disc_in"]     = w.shape[1]
                info["disc_hidden"] = w.shape[0]
                break

    disc_weights = sorted([k for k in keys
                           if "env_disc_ee" in k and "weight" in k
                           and state_dict[k].dim() == 2])
    if disc_weights:
        info["disc_out"] = state_dict[disc_weights[-1]].shape[0]

    info["has_proj"] = any("proj_head" in k for k in keys)
    if info["has_proj"]:
        for k in keys:
            if "proj_head" in k and "weight" in k:
                w = state_dict[k]
                if w.dim() == 2:
                    info["proj_dim"] = w.shape[0]
                    break
    return info



def build_from_arch(arch, cfg, b_cfg, device):
    if arch["type"] == "baseline":
        from models import BaselineSpeakerModel
        return BaselineSpeakerModel(
            b_cfg, arch.get("num_spk", 175)).to(device), "baseline"

    import copy
    patched = copy.deepcopy(cfg)
    patched["model"]["spk_dim"]       = arch.get("spk_dim",      96)
    patched["model"]["env_dim"]       = arch.get("env_dim",       96)
    patched["model"]["embedding_dim"] = arch.get("emb_dim",      192)
    patched["model"]["disc_hidden"]   = arch.get("disc_hidden",  128)
    patched["model"]["disc_out"]      = arch.get("disc_out",      64)
    if "proj_dim" in arch:
        patched["model"]["proj_dim"]  = arch["proj_dim"]

    from models import DisentangledSpeakerModel
    return DisentangledSpeakerModel(
        patched,
        arch.get("num_spk", 175),
        use_proj=arch.get("has_proj", False)).to(device), "disentangled"



def shape_safe_load(model, state_dict):
    current = model.state_dict()
    matched = {k: v for k, v in state_dict.items()
               if k in current and v.shape == current[k].shape}
    skipped = [k for k in state_dict if k not in matched]
    model.load_state_dict(matched, strict=False)
    return len(matched), len(state_dict), skipped


def remap_baseline_keys(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if "num_batches_tracked" in k: continue
        nk = k
        nk = nk.replace("backbone.conv0.0.", "encoder.layer1.")
        nk = nk.replace("backbone.conv0.1.", "encoder.bn1.")
        nk = nk.replace("backbone.layer1.",  "encoder.layer2.")
        nk = nk.replace("backbone.layer2.",  "encoder.layer3.")
        nk = nk.replace("backbone.layer3.",  "encoder.layer4.")
        nk = re.sub(r"\.convs\.(\d+)\.0\.", r".res2.convs.\1.", nk)
        nk = re.sub(r"\.convs\.(\d+)\.1\.", r".res2.bns.\1.",   nk)
        nk = re.sub(r"(encoder\.layer\d+)\.bn\.", r"\1.bn2.", nk)
        nk = nk.replace("backbone.mfa.0.",       "encoder.cat_conv.")
        nk = nk.replace("backbone.mfa.1.",        "encoder.bn_cat.")
        nk = nk.replace("backbone.pool.attn.0.", "encoder.pool.attn.0.")
        nk = nk.replace("backbone.pool.attn.2.", "encoder.pool.attn.2.")
        nk = nk.replace("backbone.bn_pool.",     "encoder.bn_pool.")
        nk = nk.replace("backbone.proj.",        "encoder.fc.")
        nk = nk.replace("backbone.bn_emb.",      "encoder.bn_emb.")
        new_state[nk] = v
    return new_state



def load_any_checkpoint(ckpt_path, cfg, b_cfg, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"]
    arch  = sniff_arch(state)

    print(f"  Epoch      : {ckpt.get('epoch', '?')}")
    print(f"  Val EER    : {ckpt.get('val_eer', 'N/A')}")
    print(f"  Arch       : {arch}")

    model, mode = build_from_arch(arch, cfg, b_cfg, device)

    if arch["type"] == "baseline":
        remapped = remap_baseline_keys(state)
        n_m, n_t, _ = shape_safe_load(model, remapped)
        print(f"  Load(remap): {n_m}/{n_t} keys matched")
        if n_m < n_t * 0.5:
            n_m2, n_t2, _ = shape_safe_load(model, state)
            print(f"  Load(raw)  : {n_m2}/{n_t2} keys matched")
    else:
        n_m, n_t, skipped = shape_safe_load(model, state)
        print(f"  Load       : {n_m}/{n_t} keys matched")
        if skipped:
            print(f"  Skipped    : {skipped[:4]}"
                  f"{'...' if len(skipped)>4 else ''}")

    return model, mode, arch


def extract_embeddings(model, loader, device, mode, max_batches=500):
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for i, (mel, labels) in enumerate(loader):
            if i >= max_batches: break
            mel = mel.to(device)
            emb = (model(mel) if mode == "baseline"
                   else model.get_embedding(mel))
            all_emb.append(F.normalize(emb.float(), dim=1).cpu())
            all_lbl.append(labels if isinstance(labels, torch.Tensor)
                           else torch.tensor(labels))
    return torch.cat(all_emb), torch.cat(all_lbl)


def build_trials(emb, lbl, n_trials=10000, seed=42):
    rng = np.random.default_rng(seed)
    N   = len(lbl)
    scores, gt = [], []
    for _ in range(n_trials):
        i, j = rng.integers(0, N, size=2)
        while i == j:
            j = int(rng.integers(0, N))
        scores.append(float(F.cosine_similarity(
            emb[i].unsqueeze(0), emb[j].unsqueeze(0))))
        gt.append(int(lbl[i] == lbl[j]))
    return np.array(scores), np.array(gt)


def check_checkpoint_diff(p1, p2):
    if not (os.path.exists(p1) and os.path.exists(p2)):
        return
    d = torch.load(p1, map_location="cpu")["model_state"]
    i = torch.load(p2, map_location="cpu")["model_state"]
    common = [k for k in d
              if k in i
              and d[k].shape == i[k].shape
              and d[k].is_floating_point()]    
    if not common:
        print("\n  ✓ Checkpoints have different architectures — truly distinct")
        return
    diff = sum((d[k]-i[k]).abs().mean().item() for k in common) / len(common)
    print(f"\n  Checkpoint avg weight diff : {diff:.6f}")
    if diff < 1e-6:
        print("  ✗ IDENTICAL — improved was not saved separately")
    else:
        print(f"  ✓ Models differ (diff={diff:.4f})")



def plot_score_distributions(scores_dict, path):
    n   = len(scores_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
    for ax, (name, (sc, gt)) in zip(axes[0], scores_dict.items()):
        ax.hist(sc[gt==0], bins=60, alpha=0.6, color="steelblue",
                label="Non-target", density=True)
        ax.hist(sc[gt==1], bins=60, alpha=0.6, color="tomato",
                label="Target",     density=True)
        ax.set_title(name)
        ax.set_xlabel("Cosine Score")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ Score distributions → {path}")


def plot_tsne(emb_dict, path, n_spk=15, max_per_spk=15):
    from sklearn.manifold import TSNE
    n   = len(emb_dict)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), squeeze=False)
    for ax, (name, (emb, lbl)) in zip(axes[0], emb_dict.items()):
        uniq = lbl.unique()[:n_spk]
        mask = torch.zeros(len(lbl), dtype=torch.bool)
        for u in uniq:
            idx = (lbl==u).nonzero(as_tuple=True)[0][:max_per_spk]
            mask[idx] = True
        sub_e = emb[mask].numpy()
        sub_l = lbl[mask].numpy()
        perp  = min(30, max(2, len(sub_e)-1))
        z = TSNE(n_components=2, perplexity=perp,
                 random_state=42, max_iter=1000).fit_transform(sub_e)
        ax.scatter(z[:,0], z[:,1], c=sub_l, cmap="tab20", s=15, alpha=0.8)
        ax.set_title(f"t-SNE: {name}"); ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ t-SNE → {path}")


def plot_training_curves(path):
    files  = {"Baseline":     "results/baseline_train_history.json",
              "Disentangled": "results/disentangled_train_history.json",
              "Improved":     "results/improved_train_history.json"}
    colors = {"Baseline":"steelblue",
              "Disentangled":"tomato",
              "Improved":"seagreen"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, fpath in files.items():
        if not os.path.exists(fpath): continue
        with open(fpath) as f:
            h = json.load(f)
        ep = [x["epoch"]       for x in h]
        lo = [x["loss"]        for x in h]
        ee = [x.get("val_eer") for x in h]
        ax1.plot([ep[i] for i,v in enumerate(lo) if v==v],
                 [v     for v   in lo             if v==v],
                 label=name, color=colors[name], marker="o", linewidth=2)
        ax2.plot([ep[i] for i,v in enumerate(ee) if v],
                 [v     for v   in ee             if v],
                 label=name, color=colors[name], marker="s", linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val EER (%)")
    ax2.set_title("Validation EER (lower = better)")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle("Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ Training curves → {path}")


def plot_metric_comparison(results, path, test_tag=""):
    models = list(results.keys())
    eers   = [results[m]["eer"]    for m in models]
    dcfs   = [results[m]["mindcf"] for m in models]
    colors = ["steelblue","tomato","seagreen"][:len(models)]
    x      = np.arange(len(models))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for ax, vals, ylabel, title in [
        (ax1, eers, "EER (%) ↓",  "Equal Error Rate"),
        (ax2, dcfs, "minDCF ↓",   "Minimum DCF"),
    ]:
        bars = ax.bar(x, vals, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_ylabel(ylabel); ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()*1.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    title_tag = f" [{test_tag}]" if test_tag else ""
    plt.suptitle(f"Speaker Verification — All Models{title_tag}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ Metric comparison → {path}")


def save_results_table(results, path, test_tag=""):
    tag_line = f"  Test set : {test_tag}" if test_tag else \
               "  Test set : internal split (train-clean-100 15%)"
    lines = ["="*55,
             "  Speaker Verification Results",
             tag_line,
             "="*55,
             f"  {'Model':<22} {'EER (%)':<12} {'minDCF'}",
             "  "+"-"*51]
    for m, v in results.items():
        lines.append(f"  {m:<22} {v['eer']:<12.2f} {v['mindcf']:.4f}")
    lines.append("="*55)
    text = "\n".join(lines)
    print("\n"+text)
    with open(path, "w") as f:
        f.write(text+"\n")
    print(f"  ✓ Results table → {path}")



def evaluate_all(args):
    cfg   = load_config(args.config)
    b_cfg = load_config("configs/baseline.yaml")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    print(f"  Device    : {device}")

    if args.test_set:
        print(f"\n  Test mode : LibriSpeech {args.test_set} (external)")
        test_ds = LibriSpeechTestDataset(
            cfg,
            root=cfg["data"].get("root", "data"),
            url=args.test_set)
        test_tag = args.test_set
    else:
        print("\n  Test mode : internal 15% split from train-clean-100")
        test_ds  = LibriSpeakerDataset(cfg, split="test", download=True)
        test_tag = "train-clean-100 (15% split)"

    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=cfg["data"].get("num_workers", 1),
        pin_memory=cfg["data"].get("pin_memory", True))

    print(f"  Test speakers : {test_ds.num_speakers}")
    print(f"  Test samples  : {len(test_ds)}")

    check_checkpoint_diff(args.disentangled_ckpt, args.improved_ckpt)

    results, emb_dict, score_dict = {}, {}, {}

    def _eval(model, mode, name):
        emb, lbl = extract_embeddings(model, test_loader, device, mode)
        sc, gt   = build_trials(emb, lbl)
        eer      = compute_eer(sc, gt)
        dcf      = compute_mindcf(sc, gt)
        results[name]    = {"eer": eer, "mindcf": dcf}
        emb_dict[name]   = (emb, lbl)
        score_dict[name] = (sc, gt)
        print(f"  {name:<22}  EER={eer:.2f}%   minDCF={dcf:.4f}")

    ckpts = [
        (args.baseline_ckpt,     "Baseline",     b_cfg),
        (args.disentangled_ckpt, "Disentangled", cfg),
        (args.improved_ckpt,     "Improved",     cfg),
    ]

    for ckpt_path, name, model_cfg in ckpts:
        print(f"\n  ── {name} ────────────────────────────────────────")
        if not os.path.exists(ckpt_path):
            print(f"  ⚠  Not found: {ckpt_path}  (skipping)")
            continue
        model, mode, arch = load_any_checkpoint(
            ckpt_path, model_cfg, b_cfg, device)
        _eval(model, mode, name)

    if not results:
        print("\n  ✗ No checkpoints found."); return

    suffix = f"_{args.test_set.replace('-','_')}" if args.test_set else ""

    with open(f"results/metrics{suffix}.json","w") as f:
        json.dump(results, f, indent=2)

    save_results_table(results,
                       f"results/results_table{suffix}.txt",
                       test_tag=test_tag)
    plot_score_distributions(score_dict,
                             f"results/score_distributions{suffix}.png")
    plot_tsne(emb_dict,
              f"results/tsne{suffix}.png")
    plot_training_curves("results/training_curves.png")
    plot_metric_comparison(results,
                           f"results/metric_comparison{suffix}.png",
                           test_tag=test_tag)

    print("\n"+"="*55)
    print(f"  ✓  All outputs saved to results/  (suffix='{suffix}')")
    print("="*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="configs/disentangled.yaml")
    parser.add_argument("--baseline_ckpt",
                        default="checkpoints/baseline_best.pt")
    parser.add_argument("--disentangled_ckpt",
                        default="checkpoints/disentangled_best.pt")
    parser.add_argument("--improved_ckpt",
                        default="checkpoints/improved_best.pt")
    parser.add_argument("--test_set",
                        default=None,
                        choices=["test-clean", "test-other",
                                 "dev-clean",  "dev-other"],
                        help="Use external LibriSpeech split instead of "
                             "internal 15%% held-out split. "
                             "Auto-downloads if not present.")
    args = parser.parse_args()
    evaluate_all(args)
