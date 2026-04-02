# Q3 — Ethical Auditing & Documentation Debt Mitigation

## Hugging Face Model
https://huggingface.co/surajkumar843/wav2vec2-fairness-q3


## Setup

Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Setup environment file
cp .env.example .env

Step 3: Run the project
python main.py


## Environment Variables

Add the following inside .env file:

HF_USERNAME=your_username
HF_TOKEN=hf_your_access_token


## Configuration

All parameters are defined in config.yaml.

Important settings:

dataset:
  url: "test-clean"        # ~334MB dataset (auto-download)
  max_train_samples: 2000

training:
  model_name: "facebook/wav2vec2-base-960h"
  n_epochs: 3
  batch_size: 4
  lambda_fair: 0.5


## Project Structure

Q3/
├── main.py                        # main execution script
├── audit.py                       # Part 1: bias audit
├── privacymodule.py               # Part 2: voice transformation
├── pp_demo.py                     # Part 2: demo generation
├── train_fair.py                  # Part 3: fairness-aware training
├── evaluation_scripts/
│   └── fad_eval.py                # Part 4: evaluation
├── config.yaml
├── config_loader.py
├── .env
├── requirements.txt

├── Results/examples/
│   └── examples/
│       ├── pair_XXX_orig_M.wav
│       └── pair_XXX_trans_M2F.wav

└── Results/output/
    ├── audit_plots.png / .pdf
    ├── speakers_audit.csv
    ├── gender_time_audit.csv
    ├── documentation_debt.csv
    ├── spectrogram_pairs.png
    ├── transformation_metadata.csv
    ├── training_results.png
    ├── training_losses.csv
    ├── wer_by_gender.csv
    └── evaluation/
        ├── fad_dnsmos_validation.png
        ├── validation_results.csv
        ├── dnsmos_original.csv
        └── dnsmos_transformed.csv


## Model Checkpoint

Best model is saved at:
./Results/best_model/

Also uploaded to:
{HF_USERNAME}/wav2vec2-fairness-q3

Training details:
- Dataset: test-clean
- Training samples: 2000
- Epochs: 3
- Fairness weight (lambda): 0.5
- Base model: wav2vec2-base-960h
- Feature encoder is frozen

Checkpoint is selected based on lowest:
CTC Loss + Fairness Loss


## Results

All outputs are stored in:
Results/

Final evaluation file:
Results/output/evaluation/validation_results.csv

This includes:
- FAD score
- DNSMOS score
- SNR values


## Summary

This project includes:
- Bias auditing of speech data
- Privacy-preserving voice transformation
- Fairness-aware ASR training
- Evaluation using audio quality metrics