"""
config_loader.py — Loads config.yaml and .env, returns a single CFG dict.
Safe to import even before pyyaml is pip-installed by catching ImportError.
"""

import os

def load_config(path: str = "config.yaml") -> dict:
    try:
        import yaml
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "python-dotenv"])
        import yaml

    try:
        from dotenv import load_dotenv
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "python-dotenv"])
        from dotenv import load_dotenv

    load_dotenv()

    # Resolve path relative to this file's directory, not cwd
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, path) if not os.path.isabs(path) else path

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"config.yaml not found at '{cfg_path}'. "
            f"Make sure config.yaml is in the same directory as config_loader.py"
        )

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Inject HuggingFace secrets from .env
    if "huggingface" not in cfg:
        cfg["huggingface"] = {}
    cfg["huggingface"]["username"] = os.getenv("HF_USERNAME", "")
    cfg["huggingface"]["token"]    = os.getenv("HF_TOKEN", "")

    return cfg


# Module-level singleton — loaded once on first import
CFG = load_config()
