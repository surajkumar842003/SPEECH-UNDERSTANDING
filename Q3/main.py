"""
main.py — Master runner for Q3. Installs all deps first, THEN imports CFG.
"""

import subprocess, sys, os, time

# ── STEP 0: Install ALL deps before any other import ─────────
print("=" * 65)
print("Q3: Ethical Auditing & Documentation Debt Mitigation")
print("Installing dependencies ...")
print("=" * 65)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "torch", "torchaudio", "transformers", "librosa", "soundfile",
    "numpy", "pandas", "matplotlib", "seaborn", "scipy",
    "jiwer", "huggingface_hub", "python-dotenv", "tqdm", "pyyaml"
])
print("All dependencies installed.\n")

# ── STEP 1: Now safe to load config ──────────────────────────
from config_loader import CFG

# Validate all expected keys exist
def _require(cfg, *keys):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise KeyError(
                f"Missing key in config.yaml: {' -> '.join(keys)}\n"
                f"  Got keys at '{keys[list(keys).index(k)-1] if list(keys).index(k)>0 else 'root'}': "
                f"{list(node.keys()) if isinstance(node, dict) else 'N/A'}"
            )
        node = node[k]
    return node

HF_USERNAME = _require(CFG, "huggingface", "username")
HF_TOKEN    = _require(CFG, "huggingface", "token")
MODEL_DIR   = _require(CFG, "training",    "model_save_dir")
REPO_NAME   = _require(CFG, "huggingface", "repo_name")
PRIVATE     = _require(CFG, "huggingface", "private")

# Create all output dirs upfront
for d in [_require(CFG, "output", "output_dir"),
          _require(CFG, "output", "examples_dir"),
          _require(CFG, "output", "eval_dir"),
          MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Config loaded. Dataset: '{CFG['dataset']['url']}' | "
      f"Train samples: {CFG['dataset']['max_train_samples']} | "
      f"Device will be auto-detected.\n")

# ── PART 1: Bias Audit ────────────────────────────────────────
print("═" * 65)
print("PART 1: Bias Identification & Documentation Debt Audit")
print("═" * 65)
t0 = time.time()
from audit import download_dataset, parse_speakers_txt, run_audit
dataset  = download_dataset()
spk_df   = parse_speakers_txt()
train_df, sample_indices, durations = run_audit(dataset, spk_df)
print(f"Part 1 complete in {time.time()-t0:.1f}s\n")

# ── PART 2: Privacy-Preserving Demo ──────────────────────────
print("═" * 65)
print("PART 2: Privacy-Preserving Voice Transformation Demo")
print("═" * 65)
t0 = time.time()
from pp_demo import run_demo
meta_df = run_demo()
print(f"Part 2 complete in {time.time()-t0:.1f}s\n")

# ── PART 3: Fairness Training ─────────────────────────────────
print("═" * 65)
print("PART 3: Fairness-Aware ASR Training Loop")
print("═" * 65)
t0 = time.time()
from train_fair import run_training
model, processor, wer_df = run_training()
print(f"Part 3 complete in {time.time()-t0:.1f}s\n")

# ── PART 4: FAD / DNSMOS Validation ──────────────────────────
print("═" * 65)
print("PART 4: FAD & DNSMOS Proxy Validation")
print("═" * 65)
t0 = time.time()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "evaluation_scripts"))
from fad_eval import run_evaluation
val_summary = run_evaluation()
print(f"Part 4 complete in {time.time()-t0:.1f}s\n")

# ── HuggingFace Upload ────────────────────────────────────────
print("═" * 65)
print("HUGGINGFACE MODEL UPLOAD")
print("═" * 65)
if HF_USERNAME and HF_TOKEN:
    try:
        from huggingface_hub import HfApi
        api       = HfApi()
        full_repo = f"{HF_USERNAME}/{REPO_NAME}"
        api.create_repo(
            repo_id=full_repo, token=HF_TOKEN,
            repo_type="model", private=PRIVATE, exist_ok=True
        )
        api.upload_folder(
            folder_path=MODEL_DIR,
            repo_id=full_repo,
            token=HF_TOKEN,
            repo_type="model",
            commit_message="Upload fairness-trained Wav2Vec2 (Q3)"
        )
        print(f"✓ Model uploaded → https://huggingface.co/{full_repo}")
    except Exception as e:
        print(f"✗ HuggingFace upload failed: {e}")
        print("  Check HF_USERNAME and HF_TOKEN in your .env file.")
else:
    print("⚠  HF_USERNAME / HF_TOKEN not set in .env — skipping upload.")

# ── Final Summary ─────────────────────────────────────────────
out  = CFG["output"]["output_dir"]
eval_d = CFG["output"]["eval_dir"]
ex   = CFG["output"]["examples_dir"]

