import os, random, yaml
import numpy as np
import torch


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0 * 100.0)
    return eer


def compute_mindcf(scores: np.ndarray,
                   labels: np.ndarray,
                   p_target: float = 0.05,
                   c_miss:   float = 1.0,
                   c_fa:     float = 1.0) -> float:
    
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr         = 1.0 - tpr

    dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    min_dcf = float(np.min(dcf))

    beta     = c_fa * (1.0 - p_target) / (c_miss * p_target)
    norm_dcf = min(1.0, beta) if beta < 1.0 else 1.0
    min_dcf  = min_dcf / (c_miss * p_target * norm_dcf + 1e-12)

    return min_dcf
