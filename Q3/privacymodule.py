"""
privacymodule.py — Privacy-Preserving Voice Biometric Obfuscation (PyTorch)
Parameters read from config.yaml via config_loader.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

from config_loader import CFG

SR             = CFG["audio"]["sample_rate"]
N_FFT          = CFG["audio"]["n_fft"]
HOP_LENGTH     = CFG["audio"]["hop_length"]
N_MELS         = CFG["audio"]["n_mels"]
PITCH_STEPS    = CFG["privacy"]["pitch_shift_steps"]
BINS_OCT       = CFG["privacy"]["bins_per_octave"]


class VoiceBiometricObfuscator(nn.Module):
    """
    Obfuscates gender/age biometric traits while preserving linguistic content.
    Pipeline: pitch shift (phase vocoder) → spectral envelope warp → gain norm
    """

    def __init__(self):
        super().__init__()
        self.sample_rate  = SR
        self.n_fft        = N_FFT
        self.hop_length   = HOP_LENGTH
        self.n_mels       = N_MELS
        self.pitch_steps  = PITCH_STEPS
        self.bins_per_oct = BINS_OCT

        # Learnable parameters
        self.formant_scale = nn.Parameter(torch.ones(self.n_mels))
        self.gain          = nn.Parameter(torch.ones(1))

    def _to_mono_numpy(self, waveform: torch.Tensor) -> np.ndarray:
        wav = waveform.float()
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        elif wav.dim() == 2:
            wav = wav.squeeze(0)
        wav_np = wav.numpy()
        peak = np.abs(wav_np).max()
        if peak > 1e-8:
            wav_np = wav_np / peak
        return wav_np.astype(np.float32)

    def _pitch_shift(self, wav_np: np.ndarray, n_steps: float) -> np.ndarray:
        shifted = librosa.effects.pitch_shift(
            wav_np, sr=self.sample_rate,
            n_steps=n_steps, bins_per_octave=self.bins_per_oct
        )
        return shifted.astype(np.float32)

    def _spectral_envelope_warp(self, wav_np: np.ndarray, direction: str) -> np.ndarray:
        D   = librosa.stft(wav_np, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(D)
        ph  = np.angle(D)

        mel_fb   = librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels)
        scale_np = self.formant_scale.detach().cpu().numpy()

        if direction == "male_to_female":
            freq_env = np.linspace(0.85, 1.25, self.n_mels) * scale_np
        else:
            freq_env = np.linspace(1.25, 0.85, self.n_mels) * scale_np

        freq_env  = np.clip(freq_env, 0.5, 2.5)
        stft_env  = mel_fb.T @ freq_env
        stft_env  = stft_env / (stft_env.max() + 1e-8) + 0.5
        stft_env  = stft_env[:, np.newaxis]

        D_warped  = (mag * stft_env) * np.exp(1j * ph)
        out       = librosa.istft(D_warped, hop_length=self.hop_length, length=len(wav_np))
        return out.astype(np.float32)

    def forward(self, waveform: torch.Tensor, direction: str = "male_to_female") -> torch.Tensor:
        original_len = waveform.shape[-1]
        wav_np       = self._to_mono_numpy(waveform)

        n_steps     = self.pitch_steps if direction == "male_to_female" else -self.pitch_steps
        wav_shifted = self._pitch_shift(wav_np, n_steps)

        # Enforce original length
        if len(wav_shifted) > len(wav_np):
            wav_shifted = wav_shifted[:len(wav_np)]
        elif len(wav_shifted) < len(wav_np):
            wav_shifted = np.pad(wav_shifted, (0, len(wav_np) - len(wav_shifted)))

        wav_warped = self._spectral_envelope_warp(wav_shifted, direction)

        if len(wav_warped) > original_len:
            wav_warped = wav_warped[:original_len]
        elif len(wav_warped) < original_len:
            wav_warped = np.pad(wav_warped, (0, original_len - len(wav_warped)))

        out  = torch.from_numpy(wav_warped).float().unsqueeze(0)
        gain = float(self.gain.detach().clamp(0.5, 2.0).item())
        out  = out * gain

        peak = out.abs().max().item()
        if peak > 1e-8:
            out = out / peak
        return out


def load_audio_16k(path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if sr != SR:
        waveform = T.Resample(orig_freq=sr, new_freq=SR)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def save_audio_wav(waveform: torch.Tensor, path: str):
    """Save as proper 16-bit PCM WAV — no garbage output."""
    wav_np = waveform.squeeze().float().numpy()
    peak   = np.abs(wav_np).max()
    if peak > 1e-8:
        wav_np = wav_np / peak
    wav_i16 = (wav_np * 32767).astype(np.int16)
    sf.write(path, wav_i16, SR, subtype="PCM_16")
