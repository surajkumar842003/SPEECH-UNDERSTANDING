import argparse, os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloader, LibriSpeakerDataset
from models  import BaselineSpeakerModel, DisentangledSpeakerModel, grad_reverse
from losses  import (reconstruction_loss, triplet_env_loss, mapc_loss,
                     contrastive_speaker_loss, total_disentangled_loss)
from utils   import load_config, set_seed, save_checkpoint, compute_eer


def grl_alpha(epoch, total_epochs):
    return float(epoch - 1) / max(total_epochs - 1, 1)

def safe(t, name=""):
    if torch.isnan(t) or torch.isinf(t):
        if name: print(f"    ✗ NaN/Inf in {name}")
        return False
    return True

def cfg_loss(cfg, *keys, default=0.0):
    d = cfg.get("loss", {})
    for k in keys:
        if k in d: return d[k]
    return default

def cfg_train(cfg, key, default):
    return cfg.get("training", {}).get(key, default)

def cfg_imp(cfg, key, default):
    return cfg.get("improvement", {}).get(key, default)

def contrast_lambda_warmup(epoch, total, lam_max):
    warmup = min(2, total)
    return lam_max * min(1.0, epoch / warmup)

def safe_warmstart(model, warmstart_path, device):
    if not (warmstart_path and os.path.exists(warmstart_path)):
        if warmstart_path:
            print(f"  ⚠  warmstart not found: {warmstart_path}")
        return float("inf")

    print(f"\n  ♻  Warm-starting from: {warmstart_path}")
    ckpt        = torch.load(warmstart_path, map_location=device)
    saved_state = ckpt["model_state"]
    model_state = model.state_dict()

    filtered = {
        k: v for k, v in saved_state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    skipped = [k for k in saved_state
               if k not in model_state or
               saved_state[k].shape != model_state.get(k, torch.tensor([])).shape]

    model.load_state_dict(filtered, strict=False)
    print(f"  ♻  Loaded  : {len(filtered)}/{len(saved_state)} keys")
    if skipped:
        print(f"  ♻  Skipped (shape mismatch / new): {skipped[:6]}"
              f"{'...' if len(skipped)>6 else ''}")
    base_eer = float(ckpt.get("val_eer", float("inf")))
    print(f"  ♻  Base EER: {base_eer:.2f}%")
    return base_eer



def compute_val_eer(model, val_loader, device, mode="baseline"):
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for mel, labels in val_loader:
            mel = mel.to(device)
            emb = model(mel) if mode == "baseline" else model.get_embedding(mel)
            all_emb.append(F.normalize(emb.float(), dim=1).cpu())
            all_lbl.append(labels)
    all_emb = torch.cat(all_emb)
    all_lbl = torch.cat(all_lbl)
    rng = np.random.default_rng(42)
    N   = len(all_lbl)
    n_t = min(3000, N * (N-1) // 2)
    scores, gt = [], []
    for _ in range(n_t):
        i, j = rng.integers(0, N, size=2)
        while i == j:
            j = int(rng.integers(0, N))
        scores.append(float(F.cosine_similarity(
            all_emb[i].unsqueeze(0), all_emb[j].unsqueeze(0))))
        gt.append(int(all_lbl[i] == all_lbl[j]))
    return compute_eer(np.array(scores), np.array(gt))


def train_baseline(cfg, device):
    print("\n"+"="*65)
    print("  TRAINING: Baseline ECAPA-TDNN"); print("="*65)

    train_loader, num_spk = get_dataloader(cfg, split="train", triplet=False)
    val_ds     = LibriSpeakerDataset(cfg, split="val", download=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=cfg["data"].get("num_workers",1),
                            pin_memory=cfg["data"].get("pin_memory",True))

    epochs    = cfg_train(cfg, "epochs",       10)
    patience  = cfg_train(cfg, "patience",      3)
    min_delta = cfg_train(cfg, "min_delta",   0.1)
    log_int   = cfg_train(cfg, "log_interval", 10)
    lr        = cfg_train(cfg, "lr",         5e-4)
    wd        = cfg_train(cfg, "weight_decay",1e-4)
    save_dir  = cfg_train(cfg, "save_dir", "checkpoints")

    print(f"  Train spk={num_spk}  Val spk={val_ds.num_speakers}")
    model     = BaselineSpeakerModel(cfg, num_spk).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler()
    best_eer, patience_ctr, history = float("inf"), 0, []

    for epoch in range(1, epochs+1):
        model.train()
        total, n, t0 = 0.0, 0, time.time()
        for step, (mel, labels) in enumerate(train_loader):
            mel, labels = mel.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                loss, _ = model.compute_loss(mel, labels)
            if not safe(loss): continue
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            total += loss.item(); n += 1
            if (step+1) % log_int == 0:
                print(f"  [E{epoch}|{step+1}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}")
        if n == 0:
            print(f"  ✗ All NaN epoch {epoch}"); break
        scheduler.step()
        avg_loss = total / n
        val_eer  = compute_val_eer(model, val_loader, device, "baseline")
        print(f"\n  ── Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | "
              f"val_EER={val_eer:.2f}% | best={best_eer:.2f}% | "
              f"{time.time()-t0:.0f}s")
        history.append({"epoch":epoch,"loss":avg_loss,"val_eer":val_eer})
        if val_eer < best_eer - min_delta:
            best_eer, patience_ctr = val_eer, 0
            save_checkpoint(
                {"epoch":epoch,"model_state":model.state_dict(),
                 "best_eer":best_eer,"val_eer":val_eer},
                os.path.join(save_dir,"baseline_best.pt"))
            print(f"  ✓ Saved baseline_best.pt  EER={best_eer:.2f}%")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("  ⚡ Early stopping."); break

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_train_history.json","w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  ✓ Baseline done. Best val_EER={best_eer:.2f}%")


def train_disentangled(cfg, device, use_improvement=False):
    tag   = "improved"                if use_improvement else "disentangled"
    label = "Improved (+Contrastive)" if use_improvement else "Disentangled AE"

    train_loader, num_spk = get_dataloader(cfg, split="train", triplet=True)
    val_ds     = LibriSpeakerDataset(cfg, split="val", download=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=cfg["data"].get("num_workers",1),
                            pin_memory=cfg["data"].get("pin_memory",True))

    epochs    = cfg_train(cfg, "epochs",       10)
    patience  = cfg_train(cfg, "patience",      5)
    min_delta = cfg_train(cfg, "min_delta",   0.05)
    log_int   = cfg_train(cfg, "log_interval", 10)
    lr        = cfg_train(cfg, "lr",         5e-4)
    wd        = cfg_train(cfg, "weight_decay",1e-4)
    save_dir  = cfg_train(cfg, "save_dir", "checkpoints")

    lam_spk  = cfg_loss(cfg, "lambda_spk",     default=1.0)
    lam_r    = cfg_loss(cfg, "lambda_recons",  default=0.5)
    lam_ee   = cfg_loss(cfg, "lambda_env_env", default=1.0)
    lam_adv  = cfg_loss(cfg, "lambda_adv",     default=0.3)
    lam_c    = cfg_loss(cfg, "lambda_corr",    default=0.3)
    t_margin = cfg_loss(cfg, "triplet_margin", "margin", default=0.2)

    use_contrast = cfg_imp(cfg, "use_contrastive", False) and use_improvement
    cont_temp    = float(cfg_imp(cfg, "contrastive_temp", 0.1))
    lam_contrast = float(cfg_imp(cfg, "lambda_contrast",  0.5))

    print(f"  Train spk={num_spk}  Val spk={val_ds.num_speakers}")
    print(f"  Epochs={epochs}  patience={patience}  lr={lr}")
    print(f"  Contrastive={use_contrast}  temp={cont_temp}  lambda_max={lam_contrast}")

    model = DisentangledSpeakerModel(
        cfg, num_spk, use_proj=use_improvement).to(device)

    warmstart   = cfg.get("training", {}).get("warmstart_ckpt", None)
    initial_eer = safe_warmstart(model, warmstart, device)

    disc_params = (list(model.env_disc_ee.parameters()) +
                   list(model.env_disc_es.parameters()))
    main_params = (list(model.backbone.parameters()) +
                   list(model.ae.parameters())       +
                   list(model.spk_cls.parameters()))
    if model.proj_head is not None:
        main_params += list(model.proj_head.parameters())

    opt_main  = Adam(main_params, lr=lr,       weight_decay=wd)
    opt_disc  = Adam(disc_params, lr=lr * 0.1)
    scheduler = CosineAnnealingLR(opt_main, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler()

    best_eer     = initial_eer
    patience_ctr = 0
    history      = []

    for epoch in range(1, epochs+1):
        model.train()
        total, n, t0 = 0.0, 0, time.time()
        alpha   = grl_alpha(epoch, epochs)
        lam_con = contrast_lambda_warmup(epoch, epochs, lam_contrast)
        print(f"\n  Epoch {epoch}/{epochs}  "
              f"GRL_alpha={alpha:.3f}  lam_contrast={lam_con:.3f}")

        for step, (mel1, mel2, mel3, labels) in enumerate(train_loader):
            mel1   = mel1.to(device)
            mel2   = mel2.to(device)
            mel3   = mel3.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                espk1, eenv1, ehat1, e1, z1 = model(mel1)
                espk2, eenv2, ehat2, e2, z2 = model(mel2)
                espk3, eenv3, ehat3, e3, z3 = model(mel3)

                espk1 = F.normalize(espk1.float(), dim=1)
                espk2 = F.normalize(espk2.float(), dim=1)
                espk3 = F.normalize(espk3.float(), dim=1)
                eenv1 = F.normalize(eenv1.float(), dim=1)
                eenv2 = F.normalize(eenv2.float(), dim=1)
                eenv3 = F.normalize(eenv3.float(), dim=1)

                loss_r = (reconstruction_loss(e1, ehat1) +
                          reconstruction_loss(e2, ehat2) +
                          reconstruction_loss(e3, ehat3)) / 3

                loss_spk = model.spk_cls(
                    torch.cat([espk1, espk2, espk3], dim=0),
                    labels.repeat(3))

                loss_ee = triplet_env_loss(
                    model.env_disc_ee(eenv1),
                    model.env_disc_ee(eenv2),
                    model.env_disc_ee(eenv3), t_margin)

                loss_adv = triplet_env_loss(
                    model.env_disc_es(grad_reverse(espk1, alpha)),
                    model.env_disc_es(grad_reverse(espk2, alpha)),
                    model.env_disc_es(grad_reverse(espk3, alpha)), t_margin)

                loss_corr = (mapc_loss(espk1, eenv1) +
                             mapc_loss(espk2, eenv2) +
                             mapc_loss(espk3, eenv3)) / 3

            loss_contrast = None
            if use_contrast and z1 is not None:
                loss_contrast = contrastive_speaker_loss(
                    [F.normalize(z1.float(), dim=1),
                     F.normalize(z2.float(), dim=1),
                     F.normalize(z3.float(), dim=1)],
                    labels, temperature=cont_temp)
                if not safe(loss_contrast, "contrast"):
                    loss_contrast = None

            if not all(safe(l) for l in
                       [loss_r, loss_spk, loss_ee, loss_adv, loss_corr]):
                continue

            loss = total_disentangled_loss(
                loss_spk, loss_r, loss_ee, loss_adv, loss_corr,
                loss_contrast,
                lam_spk=lam_spk, lam_r=lam_r,
                lam_ee=lam_ee,   lam_adv=lam_adv,
                lam_c=lam_c,     lam_contrast=lam_con)

            if not safe(loss): continue

            opt_main.zero_grad(); opt_disc.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt_main); scaler.unscale_(opt_disc)
            torch.nn.utils.clip_grad_norm_(main_params, 5.0)
            torch.nn.utils.clip_grad_norm_(disc_params, 1.0)
            scaler.step(opt_main); scaler.step(opt_disc)
            scaler.update()
            total += loss.item(); n += 1

            if (step+1) % log_int == 0:
                cv = loss_contrast.item() if loss_contrast is not None else 0.0
                print(f"  [E{epoch}|{step+1}/{len(train_loader)}] "
                      f"total={loss.item():.4f}  spk={loss_spk.item():.3f}  "
                      f"recon={loss_r.item():.4f}  corr={loss_corr.item():.5f}  "
                      f"adv={loss_adv.item():.4f}  contrast={cv:.4f}")

        if n == 0:
            print(f"  ✗ All NaN epoch {epoch}"); break

        scheduler.step()
        avg_loss = total / n
        val_eer  = compute_val_eer(model, val_loader, device, "disentangled")
        print(f"\n  ── Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | "
              f"val_EER={val_eer:.2f}% | best={best_eer:.2f}% | "
              f"alpha={alpha:.3f} | {time.time()-t0:.0f}s")

        history.append({"epoch":epoch,"loss":avg_loss,
                        "val_eer":val_eer,"grl_alpha":alpha})

        if val_eer < best_eer - min_delta:
            best_eer, patience_ctr = val_eer, 0
            save_checkpoint(
                {"epoch":epoch,"model_state":model.state_dict(),
                 "best_eer":best_eer,"val_eer":val_eer},
                os.path.join(save_dir, f"{tag}_best.pt"))
            print(f"  ✓ Saved {tag}_best.pt  EER={best_eer:.2f}%")
        else:
            patience_ctr += 1
            print(f"  No improvement. Patience {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print("  ⚡ Early stopping."); break

    os.makedirs("results", exist_ok=True)
    with open(f"results/{tag}_train_history.json","w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  ✓ {label} done. Best val_EER={best_eer:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode",
                        choices=["baseline","disentangled","improved"],
                        default="baseline")
    args   = parser.parse_args()
    cfg    = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    os.makedirs(cfg.get("training",{}).get("save_dir","checkpoints"),
                exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if args.mode == "baseline":
        train_baseline(cfg, device)
    elif args.mode == "disentangled":
        train_disentangled(cfg, device, use_improvement=False)
    elif args.mode == "improved":
        train_disentangled(cfg, device, use_improvement=True)
