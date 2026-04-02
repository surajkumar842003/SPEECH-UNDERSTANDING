# Q3 — Speaker Verification with Disentangled ECAPA-TDNN

This project implements a speaker verification system based on ECAPA-TDNN.
It introduces disentangled representations to separate speaker identity
from environmental factors. All experiments are conducted using the
LibriSpeech train-clean-100 dataset.


## Available Models

| Model File              | Details                                           |
|------------------------|--------------------------------------------------|
| baseline_best.pt       | Standard ECAPA-TDNN with AAM-Softmax loss        |
| disentangled_best.pt   | Adds disentanglement using AE + GRL + MAPC       |
| improved_best.pt       | Further enhanced using SimCLR (NT-Xent loss)     |


## Installation

Install required libraries using:

pip install torch torchaudio scikit-learn matplotlib huggingface_hub


## Training Procedure

Run the models in the following sequence:

# 1. Train baseline model
python train.py --config configs/baseline.yaml --mode baseline

# 2. Train disentangled model
python train.py --config configs/disentangled.yaml --mode disentangled

# 3. Fine-tune improved model (uses previous checkpoint)
python train.py --config configs/improved_finetune.yaml --mode improved


## Running Training in Background

mkdir -p logs

nohup python train.py --config configs/baseline.yaml --mode baseline \
> logs/baseline.log 2>&1 &

# Monitor logs
tail -f logs/baseline.log


## Evaluation

# Evaluate on internal validation split (default ~15%)
python eval.py

# Evaluate on LibriSpeech test-clean dataset (auto-download)
python eval.py --test_set test-clean


## Output Files

All evaluation results are stored in the results/ directory:

results_table.txt          → EER and minDCF values  
metric_comparison.png      → comparison plot  
score_distributions.png    → histogram of scores  
tsne.png                   → visualization of embeddings  
training_curves.png        → training loss and EER trends  


## Project Directory Structure

Q3/
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