
# Q3. Speaker Verification — Disentangled ECAPA-TDNN

Speaker verification using ECAPA-TDNN with disentanglement of speaker
and environment embeddings. Three models trained on LibriSpeech train-clean-100.


## Models

| Checkpoint              | Description                                      |
|-------------------------|--------------------------------------------------|
| `baseline_best.pt`      | ECAPA-TDNN + AAM-Softmax                        |
| `disentangled_best.pt`  | ECAPA-TDNN + DisentanglerAE + GRL + MAPC        |
| `improved_best.pt`      | Disentangled + SimCLR NT-Xent on E_S            |

## Setup

```bash
pip install torch torchaudio scikit-learn matplotlib huggingface_hub
Training
bash
# Step 1 — Baseline
python train.py --config configs/baseline.yaml --mode baseline

# Step 2 — Disentangled
python train.py --config configs/disentangled.yaml --mode disentangled

# Step 3 — Improved (warm-starts from disentangled checkpoint)
python train.py --config configs/improved_finetune.yaml --mode improved
Run in background:

bash
mkdir -p logs
nohup python train.py --config configs/baseline.yaml --mode baseline \
  > logs/baseline.log 2>&1 &
tail -f logs/baseline.log
Evaluation
bash
# Internal 15% held-out split (default)
python eval.py

# External LibriSpeech test-clean (auto-downloads 334MB)
python eval.py --test_set test-clean
Results saved to results/:

results_table.txt — EER + minDCF

metric_comparison.png — bar chart

score_distributions.png — score histograms

tsne.png — speaker embedding t-SNE

training_curves.png — loss + EER curves

Project Structure
text
Q2/
├── train.py
├── eval.py
├── models.py
├── losses.py
├── dataset.py
├── utils.py
├── configs/
│   ├── baseline.yaml
│   ├── disentangled.yaml
│   └── improved_finetune.yaml
├── checkpoints/
│   ├── baseline_best.pt
│   ├── disentangled_best.pt
│   └── improved_best.pt
└── results/
