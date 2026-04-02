import os, sys, subprocess, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
PYTHON = sys.executable


def run(cmd):
    print(f"\n{'>' * 62}\n  {cmd}\n{'>' * 62}")
    t0  = time.time()
    ret = subprocess.run(cmd, shell=True, cwd=SCRIPT_DIR)
    if ret.returncode != 0:
        print(f"\n  ✗ FAILED (exit {ret.returncode}) — fix error above.")
        sys.exit(ret.returncode)
    print(f"  ✓ Done in {time.time() - t0:.1f}s")


def main():
    for d in ["checkpoints", "results"]:
        os.makedirs(os.path.join(SCRIPT_DIR, d), exist_ok=True)

    print("\n" + "█" * 62)
    print("  Q2: Disentangled Speaker Recognition — Full Pipeline")
    print("  Epochs     : 10 max per model")
    print("  Early stop : patience=3 on val EER")
    print("  Best model : saved by lowest Validation EER")
    print("  HF upload  : python upload_to_hf.py  (run separately)")
    print("█" * 62)

    run(f'"{PYTHON}" train.py --config configs/baseline.yaml --mode baseline')
    run(f'"{PYTHON}" train.py --config configs/disentangled.yaml --mode disentangled')
    run(f'"{PYTHON}" train.py --config configs/disentangled.yaml --mode improved')
    run(f'"{PYTHON}" eval.py')

    print("\n" + "█" * 62)
    print("  ALL DONE")
    print("  Checkpoints : checkpoints/baseline_best.pt")
    print("                checkpoints/disentangled_best.pt")
    print("                checkpoints/improved_best.pt")
    print("  Results     : results/metrics.json")
    print("                results/results_table.txt")
    print("                results/metric_comparison.png")
    print("                results/training_curves.png")
    print("                results/score_distributions.png")
    print("                results/tsne.png")
    print("█" * 62)


if __name__ == "__main__":
    main()
