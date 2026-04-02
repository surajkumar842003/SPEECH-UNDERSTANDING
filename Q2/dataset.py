
import os, random
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader


def _get_data_root(cfg):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data")


def _make_mel(waveform, sr, cfg):
    data_cfg = cfg["data"]
    if sr != data_cfg["sample_rate"]:
        waveform = torchaudio.functional.resample(
            waveform, sr, data_cfg["sample_rate"])

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=data_cfg["sample_rate"],
        n_fft=data_cfg["n_fft"],
        win_length=data_cfg["win_length"],
        hop_length=data_cfg["hop_length"],
        n_mels=data_cfg["n_mels"])

    mel = mel_transform(waveform)                     
    mel = (mel + 1e-6).log()
    mel = mel.squeeze(0)                              
    return mel


def _crop_or_pad(mel, max_frames):
    T = mel.shape[-1]
    if T >= max_frames:
        start = random.randint(0, T - max_frames)
        mel   = mel[:, start : start + max_frames]
    else:
        mel = F.pad(mel, (0, max_frames - T))
    return mel


def _augment(waveform, cfg):
    aug = cfg["data"].get("augmentation", {})
    if random.random() < aug.get("noise_prob", 0.0):
        snr_min, snr_max = aug.get("snr_range", [5, 20])
        snr   = random.uniform(snr_min, snr_max)
        sig_p = waveform.pow(2).mean()
        nse_p = sig_p / (10 ** (snr / 10))
        noise = torch.randn_like(waveform) * nse_p.sqrt()
        waveform = waveform + noise
    return waveform


# ── Dataset ───────────────────────────────────────────────────────────────────

class LibriSpeakerDataset(Dataset):
    
    def __init__(self, cfg, split="train", download=True, triplet=False):
        super().__init__()
        self.cfg        = cfg
        self.split      = split
        self.triplet    = triplet
        self.max_frames = cfg["data"].get("max_frames", 300)
        self.augment    = (split == "train")

        root = _get_data_root(cfg)
        url  = cfg["data"]["subset"]          

        print(f"  [Dataset] root={root}  url={url}")
        raw = torchaudio.datasets.LIBRISPEECH(root, url=url, download=download)

        spk_to_samples = defaultdict(list)
        for i in range(len(raw)):
            meta = raw.get_metadata(i)        
            spk  = str(meta[3])
            spk_to_samples[spk].append(i)

        all_spk = sorted(spk_to_samples.keys())
        print(f"  [Dataset] speakers={len(all_spk)}  "
              f"samples={sum(len(v) for v in spk_to_samples.values())}")

        random.seed(42)
        shuffled = all_spk.copy()
        random.shuffle(shuffled)
        n        = len(shuffled)
        n_train  = int(0.70 * n)
        n_val    = int(0.15 * n)

        if split == "train":
            spk_set = set(shuffled[:n_train])
        elif split == "val":
            spk_set = set(shuffled[n_train : n_train + n_val])
        else:  
            spk_set = set(shuffled[n_train + n_val :])

        self.spk_list   = sorted(spk_set)
        self.spk_to_idx = {s: i for i, s in enumerate(self.spk_list)}
        print(f"  [Split:{split:<5}] speakers={len(self.spk_list)}  "
              f"samples={sum(len(spk_to_samples[s]) for s in self.spk_list)}")

        self.samples = []
        for spk in self.spk_list:
            idx = self.spk_to_idx[spk]
            for raw_i in spk_to_samples[spk]:
                self.samples.append((raw_i, idx))

        self.spk_to_sample_idxs = defaultdict(list)
        for pos, (_, spk_idx) in enumerate(self.samples):
            self.spk_to_sample_idxs[spk_idx].append(pos)

        self.raw        = raw
        self.num_speakers = len(self.spk_list)

    def __len__(self):
        return len(self.samples)

    def _load_mel(self, pos, augment=False):
        raw_i, spk_idx = self.samples[pos]
        wav, sr, *_    = self.raw[raw_i]
        if augment:
            wav = _augment(wav, self.cfg)
        mel = _make_mel(wav, sr, self.cfg)
        mel = _crop_or_pad(mel, self.max_frames)
        return mel, spk_idx

    def __getitem__(self, idx):
        mel, spk_idx = self._load_mel(idx, augment=self.augment)

        if not self.triplet:
            return mel, spk_idx

        same = self.spk_to_sample_idxs[spk_idx]
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(same)
        mel_pos, _ = self._load_mel(pos_idx, augment=self.augment)

        neg_spk = spk_idx
        while neg_spk == spk_idx:
            neg_spk = random.randint(0, self.num_speakers - 1)
        neg_idx = random.choice(self.spk_to_sample_idxs[neg_spk])
        mel_neg, _ = self._load_mel(neg_idx, augment=self.augment)

        return mel, mel_pos, mel_neg, torch.tensor(spk_idx)



def get_dataloader(cfg, split="train", triplet=False):
    ds = LibriSpeakerDataset(cfg, split=split, download=True, triplet=triplet)

    num_workers = cfg["data"].get("num_workers", 2)
    pin_memory  = cfg["data"].get("pin_memory",  True)
    batch_size  = cfg["training"]["batch_size"]
    shuffle     = (split == "train")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True)

    print(f"  Train speakers : {ds.num_speakers}")
    return loader, ds.num_speakers
