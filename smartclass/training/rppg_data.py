"""Step 6 - UBFC-rPPG cache -> PyTorch chunks for PhysNet, subject split, normalisation, augmentation."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import PHYS_T


def load_cache(cache_dir):
    """{subject_id: dict(frames, rgb, ppg, fps)} for every .npz of the cache."""
    out = {}
    for p in sorted(Path(cache_dir).glob("subject*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        with np.load(p) as z:
            out[p.stem] = {k: z[k] for k in ("frames", "rgb", "ppg", "fps")}
    return out


def split_subjects(ids, seed=42, n_test=8, n_val=6):
    """Train / val / test SUBJECTS (a person is never in two splits)."""
    ids = sorted(ids)
    if len(ids) < 3:
        raise ValueError(f"need at least 3 subjects, found {len(ids)}")
    n_test = min(n_test, max(1, len(ids) // 5))
    n_val = min(n_val, max(1, len(ids) // 6))
    order = [str(s) for s in np.random.default_rng(seed).permutation(ids)]
    return sorted(order[n_test + n_val:]), sorted(order[n_test:n_test + n_val]), sorted(order[:n_test])


def normalize_clip(frames):
    """uint8 (T, H, W, 3) -> float32 (3, T, H, W), zero mean / unit std per clip.
    The same function is used live (Step 6b)."""
    x = frames.astype(np.float32) / 255.0
    x = (x - x.mean()) / (x.std() + 1e-6)
    return np.ascontiguousarray(x.transpose(3, 0, 1, 2))


def zscore(v):
    return ((v - v.mean()) / (v.std() + 1e-6)).astype(np.float32)


class ChunkDataset(Dataset):
    """Overlapping PHYS_T-frame chunks of the given subjects -> (clip, ppg)."""

    def __init__(self, data, subjects, t=PHYS_T, stride=None, augment=False, seed=0):
        self.data, self.t, self.augment = data, t, augment
        stride = stride or t // 2
        self.index = [(s, i) for s in subjects for i in range(0, len(data[s]["ppg"]) - t + 1, stride)]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        s, i = self.index[k]
        frames = self.data[s]["frames"][i:i + self.t]
        ppg = self.data[s]["ppg"][i:i + self.t]
        if self.augment:
            r = self.rng
            if r.random() < 0.5:
                frames = frames[:, :, ::-1]                       # horizontal flip
            frames = np.clip(frames.astype(np.float32) * r.uniform(0.8, 1.2) + r.uniform(-15, 15), 0, 255)
        return torch.from_numpy(normalize_clip(np.asarray(frames))), torch.from_numpy(zscore(ppg))
