"""Step 3 - load the recorded gesture dataset, split it by person, PyTorch Datasets with augmentation."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import GESTURE_DIR, GESTURES, SEQ_LEN
from ..data.gesture_dataset import list_samples, load_sample
from ..data.gesture_features import MIRROR_SWAP, fix_length, mirror_x, sequence_features

CLASSES = list(GESTURES)
CLIP_FRAMES = 8     # frames per clip for the CNN+GRU (subsampled from SEQ_LEN)
CLIP_SIZE = 64      # crop size for the CNN (stored crops are 96x96)
SWAP_IDX = np.array([CLASSES.index(MIRROR_SWAP.get(c, c)) for c in CLASSES])


def clip_indices(t, n=CLIP_FRAMES):
    return np.linspace(0, t - 1, n).round().astype(int)


def load_all(root=GESTURE_DIR, with_crops=True, log=print):
    """Load every sample into memory (the dataset is small: ~18 KB per sample).

    All samples are normalised to the mirrored view (what the camera page shows by default).
    """
    lms, pres, ys, persons, crops = [], [], [], [], []
    items = list_samples(root)
    for k, (path, person, gesture) in enumerate(items):
        try:
            s = load_sample(path, with_crops=with_crops)
        except Exception as e:  # noqa: BLE001 - one broken file must not stop the training
            if log:
                log(f"  WARNING: skipped unreadable sample {person}/{gesture}/{path.name}: {type(e).__name__}")
            continue
        L, P = fix_length(s["landmarks"], s["present"], SEQ_LEN)
        mirrored = bool(s["mirror"])
        if not mirrored:
            L = mirror_x(L, P)
        lms.append(L)
        pres.append(P)
        ys.append(CLASSES.index(gesture))
        persons.append(person)
        if with_crops:
            c = s.get("crops")
            if c is None:
                crops.append(None)
            else:
                c = c[clip_indices(len(c))]
                c = np.stack([cv2.resize(f, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_AREA) for f in c])
                crops.append(c if mirrored else c[:, :, ::-1].copy())
        if log and (k + 1) % 500 == 0:
            log(f"  loaded {k + 1}/{len(items)} samples")
    data = {
        "landmarks": np.stack(lms) if lms else np.zeros((0, SEQ_LEN, 21, 3), np.float32),
        "present": np.stack(pres) if pres else np.zeros((0, SEQ_LEN), bool),
        "y": np.array(ys, np.int64),
        "persons": np.array(persons),
        "crops": None,
    }
    if with_crops and crops and all(c is not None for c in crops):
        data["crops"] = np.stack(crops)  # (N, CLIP_FRAMES, CLIP_SIZE, CLIP_SIZE, 3) RGB uint8
    return data


def split_by_person(persons, y, seed=42, test_frac=0.2, val_frac=0.15):
    """Train / val / test indices with NO person shared between splits (no identity leakage).

    With fewer than 3 people a person split is impossible: falls back to a stratified random
    split and says so in the returned note (the report must mention it).
    """
    rng = np.random.default_rng(seed)
    people = sorted({str(p) for p in persons})
    if len(people) >= 3:
        order = [str(p) for p in rng.permutation(people)]
        n_test = max(1, round(test_frac * len(people)))
        n_val = max(1, round(val_frac * len(people)))
        test_p, val_p = set(order[:n_test]), set(order[n_test:n_test + n_val])
        train_p = set(order[n_test + n_val:])
        idx = np.arange(len(persons))
        split = (idx[[p in train_p for p in persons]], idx[[p in val_p for p in persons]],
                 idx[[p in test_p for p in persons]])
        note = (f"split by person - train: {sorted(train_p)}, val: {sorted(val_p)}, test: {sorted(test_p)}")
        return split, note
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_te, n_va = max(1, round(test_frac * len(idx))), max(1, round(val_frac * len(idx)))
        te += list(idx[:n_te]); va += list(idx[n_te:n_te + n_va]); tr += list(idx[n_te + n_va:])
    note = (f"WARNING: only {len(people)} person(s) -> random split (same people in train and test, "
            "scores are optimistic). Record at least 3 people for a real person split.")
    return (np.array(tr), np.array(va), np.array(te)), note


# ---------------------------------------------------------------------------- augmentation
def augment_landmarks(L, P, y, rng):
    """Geometric + temporal augmentation on raw landmarks (frames without a hand stay zero)."""
    L = L.copy()
    if rng.random() < 0.5:                                  # mirror: right hand <-> left hand
        L = mirror_x(L, P)
        y = SWAP_IDX[y]
    if P.any():
        pts = L[P][..., :2]
        c = pts.reshape(-1, 2).mean(0)
        a = np.deg2rad(rng.uniform(-15, 15))
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
        s = rng.uniform(0.85, 1.15)
        shift = rng.uniform(-0.08, 0.08, 2)
        pts = (pts - c) @ rot.T * s + c + shift
        pts += rng.normal(0, 0.003, pts.shape)              # landmark jitter
        L[P, :, :2] = pts
    if rng.random() < 0.5:                                  # speed change: resample in time
        t = len(P)
        f = rng.uniform(0.8, 1.2)
        start = rng.uniform(0, max(0.0, t - 1 - (t - 1) * f)) if f < 1 else 0.0
        idx = np.clip(np.round(start + np.arange(t) * f), 0, t - 1).astype(int)
        L, P = L[idx], P[idx]
    return L, P, y


class SequenceDataset(Dataset):
    """Landmark sequences -> (T, F) features. Used by the MLP / LSTM / GRU."""

    def __init__(self, data, indices, augment=False, motion=True, seed=0):
        self.L, self.P, self.y = data["landmarks"], data["present"], data["y"]
        self.idx, self.augment, self.motion = np.asarray(indices), augment, motion
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        L, P, y = self.L[j], self.P[j], int(self.y[j])
        if self.augment:
            L, P, y = augment_landmarks(L, P, y, self.rng)
        return torch.from_numpy(sequence_features(L, P, self.motion)), int(y)


class ClipDataset(Dataset):
    """Hand-crop clips -> (CLIP_FRAMES, 3, S, S) float tensors. Used by the CNN+GRU."""

    def __init__(self, data, indices, augment=False, seed=0):
        self.crops, self.y = data["crops"], data["y"]
        self.idx, self.augment = np.asarray(indices), augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        clip, y = self.crops[j].astype(np.float32) / 255.0, int(self.y[j])
        if self.augment:  # the same transform for every frame of the clip
            r = self.rng
            if r.random() < 0.5:
                clip, y = clip[:, :, ::-1], int(SWAP_IDX[y])
            clip = clip * r.uniform(0.7, 1.3) + r.uniform(-0.1, 0.1)       # brightness / contrast
            dx, dy = r.integers(-4, 5, 2)
            clip = np.roll(clip, (dy, dx), axis=(1, 2))                     # small shift
        clip = (np.clip(clip, 0, 1) - 0.5) / 0.25
        return torch.from_numpy(np.ascontiguousarray(clip.transpose(0, 3, 1, 2))), y
