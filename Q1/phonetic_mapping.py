import numpy as np
import torch
import soundfile as sf
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# -----------------------------
# Load audio
# -----------------------------
data_dir = "/DATA/suraj/pr2/suraj/data"
file = [f for f in os.listdir(data_dir) if f.endswith(".wav")][0]

path = os.path.join(data_dir, file)
audio, sr = sf.read(path)

# Convert to mono
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

# Normalize
audio = audio / (np.max(np.abs(audio)) + 1e-10)

# -----------------------------
# Load model
# -----------------------------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# -----------------------------
# Forward pass
# -----------------------------
inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values).logits

pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(pred_ids[0])

print("Transcription:", transcription)

# -----------------------------
# Get model boundaries (filtered)
# -----------------------------
def get_model_boundaries(pred_ids, min_gap=20):
    pred_ids = pred_ids[0].cpu().numpy()

    boundaries = []
    prev = pred_ids[0]
    last_boundary = -min_gap

    for i, p in enumerate(pred_ids):
        if p != prev:
            if i - last_boundary >= min_gap:
                boundaries.append(i)
                last_boundary = i
            prev = p

    return np.array(boundaries)

model_boundaries = get_model_boundaries(pred_ids)

# -----------------------------
# Load manual segmentation
# -----------------------------
labels = np.load("labels.npy")

manual_boundaries = np.where(np.diff(labels) != 0)[0]

# Reduce density (important)
manual_boundaries = manual_boundaries[::3]

# -----------------------------
# Convert BOTH to time (seconds)
# -----------------------------
duration = len(audio) / sr

num_model_frames = logits.shape[1]
model_frame_time = duration / num_model_frames

model_times = model_boundaries * model_frame_time
manual_times = manual_boundaries * 0.01  # 10ms

# -----------------------------
# RMSE Calculation (aligned)
# -----------------------------
def compute_rmse(manual, model):
    if len(manual) == 0 or len(model) == 0:
        return 0.0

    n = min(len(manual), len(model))

    manual = np.interp(
        np.linspace(0, len(manual)-1, n),
        np.arange(len(manual)),
        manual
    )

    model = np.interp(
        np.linspace(0, len(model)-1, n),
        np.arange(len(model)),
        model
    )

    return np.sqrt(np.mean((manual - model) ** 2))

rmse = compute_rmse(manual_times, model_times)

print("RMSE between boundaries:", rmse)