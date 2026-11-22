"""Step 6 - UBFC-rPPG cache -> PyTorch chunks for PhysNet, subject split, normalisation, augmentation."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import PHYS_T
from ..rppg.pos import bandpass, spectrum_bpm

HR_AUG_RANGE = (45.0, 150.0)   # frequency augmentation: target heart rates (BPM), uniform
RATE_LIMITS = (0.4, 1.6)       # playback-rate limits (0.4 = 2.5x slower: 110 BPM -> 44 BPM)
RESTING_HR = 70.0              # "resting" test: videos slowed down to about this heart rate


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


@torch.no_grad()
def predict_signal(model, frames, device, t=PHYS_T, batch=8):
    """Whole video -> rPPG signal: overlapping chunks (50 %), Hann-weighted overlap-add.
    Used for the test evaluation AND live (Step 6b)."""
    model.eval()
    n = len(frames)
    starts = list(range(0, max(n - t, 0) + 1, t // 2))
    if starts and starts[-1] + t < n:
        starts.append(n - t)
    out, wsum, win = np.zeros(n), np.zeros(n), np.hanning(t) + 1e-3
    for b in range(0, len(starts), batch):
        idx = starts[b:b + batch]
        x = torch.from_numpy(np.stack([normalize_clip(frames[i:i + t]) for i in idx])).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            y = model(x).float().cpu().numpy()
        for i, yi in zip(idx, y):
            out[i:i + t] += zscore(yi) * win
            wsum[i:i + t] += win
    return out / np.maximum(wsum, 1e-9)


def resample_frames(t, frames, fs, grid=None):
    """Frames taken at irregular times t -> uniform fs grid (linear blend of the 2 nearest frames).
    Used live (webcam -> 30 fps) and by webcam_degrade (to imitate exactly that)."""
    t = np.asarray(t, dtype=np.float64)
    tu = np.arange(t[0], t[-1], 1 / fs) if grid is None else grid
    j = np.clip(np.searchsorted(t, tu, side="right") - 1, 0, len(t) - 2)
    a = ((tu - t[j]) / np.maximum(t[j + 1] - t[j], 1e-6)).clip(0, 1)[:, None, None, None]
    f = frames.astype(np.float32)
    return np.clip(f[j] * (1 - a) + f[j + 1] * a + 0.5, 0, 255).astype(np.uint8)


def webcam_degrade(frames, rng, fs=30.0, fps=(15, 30), jpeg=(50, 95), noise=(0, 3), blur=(0.3, 1.2), p_blur=0.5,
                   gain=0.08):
    """Make clean UBFC face crops look like the live webcam (Step 6b domain shift, measured live:
    PhysNet MAE 10.4 BPM vs POS 3.2). (lo, hi) = random range; lo == hi = fixed (test).
      - lower, irregular frame rate (15-30 fps, +-15 % jitter) then linear resample to 30 fps, exactly as live
      - white-balance difference (per-channel gain), sensor noise, slight blur, JPEG compression
    The PPG label is NOT changed: the pulse is the same, only the camera is worse."""
    t_len = len(frames)
    x = frames
    f_cam = rng.uniform(*fps)
    if f_cam < fs - 1:
        dt = rng.uniform(0.85, 1.15, int(t_len / fs * f_cam) + 4) / f_cam
        tc = np.concatenate([[0.0], np.cumsum(dt)]) + rng.uniform(0, 1 / f_cam)
        tc = tc[tc <= (t_len - 1) / fs]
        if len(tc) >= 2:
            x = resample_frames(tc, x[np.round(tc * fs).astype(int)], fs, grid=np.arange(t_len) / fs)
    g = rng.uniform(1 - gain, 1 + gain, 3).astype(np.float32)
    sb = rng.uniform(*blur) if rng.random() < p_blur else 0.0
    ns, q = rng.uniform(*noise), int(rng.uniform(*jpeg))
    out = np.empty_like(x)
    for i, f in enumerate(x):
        f = f.astype(np.float32) * g
        if sb > 0:
            f = cv2.GaussianBlur(f, (0, 0), sb)
        if ns > 0:
            f += rng.normal(0, ns, f.shape).astype(np.float32)
        f = np.clip(f + 0.5, 0, 255).astype(np.uint8)
        if q < 100:   # JPEG works in YCrCb from BGR: flip RGB -> BGR and back
            ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(f[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, q])
            f = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
        out[i] = f
    return out


def degrade_test_video(frames, seed):
    """Fixed, reproducible 'webcam' version of a test video: 20 fps +-15 %, JPEG 60, noise 2, blur 0.7."""
    return webcam_degrade(frames, np.random.default_rng(seed), fps=(20, 20), jpeg=(60, 60), noise=(2, 2),
                          blur=(0.7, 0.7), p_blur=1.0, gain=0.0)


def ppg_hr(ppg, fs=30.0):
    """Heart rate (BPM) of a ground-truth PPG segment."""
    return spectrum_bpm(bandpass(np.asarray(ppg, np.float64), fs), fs)[0]


def time_stretch(frames, ppg, rate, start=0.0, n_out=None, block=256):
    """Play the video AND its PPG at `rate` (0.6 = slower: a 110 BPM pulse becomes 66 BPM).
    Frames are linearly blended between neighbours, the PPG is interpolated: the label stays exact."""
    n = len(frames)
    n_out = n_out or int((n - 1 - start) / rate) + 1
    pos = start + rate * np.arange(n_out)
    j = np.clip(np.floor(pos).astype(int), 0, n - 2)
    a = (pos - j).clip(0, 1).astype(np.float32)[:, None, None, None]
    out = np.empty((n_out,) + frames.shape[1:], np.uint8)
    for b in range(0, n_out, block):   # blocks: a whole slowed video in float32 would need a lot of RAM
        sl = slice(b, b + block)
        f = frames[j[sl]].astype(np.float32) * (1 - a[sl]) + frames[j[sl] + 1].astype(np.float32) * a[sl]
        out[sl] = np.clip(f + 0.5, 0, 255).astype(np.uint8)
    return out, np.interp(pos, np.arange(n), ppg).astype(np.float32)


def resting_version(d, target=RESTING_HR):
    """Slow a whole UBFC video down so its pulse is about `target` BPM (UBFC subjects mostly have 85-120 BPM:
    they play a stressful game; the app is used at rest). Videos already <= target + 5 are kept."""
    hr = ppg_hr(d["ppg"], float(d["fps"]))
    if hr <= target + 5:
        return d
    frames, ppg = time_stretch(d["frames"], d["ppg"], target / hr)
    return {**d, "frames": frames, "ppg": ppg}


class ChunkDataset(Dataset):
    """Overlapping PHYS_T-frame chunks of the given subjects -> (clip, ppg)."""

    def __init__(self, data, subjects, t=PHYS_T, stride=None, augment=False, webcam_p=0.0, freq_p=0.0, seed=0):
        """augment: flip + brightness. webcam_p: probability of webcam_degrade (v2).
        freq_p: probability of FREQUENCY augmentation (v3): the clip is played slower/faster so that its
        heart rate becomes uniform in HR_AUG_RANGE (UBFC alone is biased towards high heart rates)."""
        self.data, self.t, self.augment, self.webcam_p, self.freq_p = data, t, augment, webcam_p, freq_p
        stride = stride or t // 2
        self.index = [(s, i) for s in subjects for i in range(0, len(data[s]["ppg"]) - t + 1, stride)]
        self.seed = seed
        # heart rate of each chunk (from the ground-truth PPG, 2x longer window for a precise value)
        self.hr = ([ppg_hr(data[s]["ppg"][max(0, i - t // 2):i + t + t // 2], float(data[s]["fps"]))
                    for s, i in self.index] if freq_p > 0 else None)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        s, i = self.index[k]
        frames = self.data[s]["frames"][i:i + self.t]
        ppg = self.data[s]["ppg"][i:i + self.t]
        if self.augment:
            # torch seeds every DataLoader worker differently each epoch -> different, reproducible augmentations
            r = np.random.default_rng([self.seed, k, int(torch.randint(0, 2 ** 31 - 1, (1,)))])
            if r.random() < self.freq_p:
                frames, ppg = self._stretched(s, i, k, r)
            if r.random() < self.webcam_p:
                frames = webcam_degrade(np.asarray(frames), r)
            if r.random() < 0.5:
                frames = frames[:, :, ::-1]                       # horizontal flip
            frames = np.clip(frames.astype(np.float32) * r.uniform(0.8, 1.2) + r.uniform(-15, 15), 0, 255)
        return torch.from_numpy(normalize_clip(np.asarray(frames))), torch.from_numpy(zscore(ppg))

    def _stretched(self, s, i, k, r):
        d, t = self.data[s], self.t
        n = len(d["ppg"])
        rate = float(np.clip(r.uniform(*HR_AUG_RANGE) / self.hr[k], *RATE_LIMITS))
        rate = min(rate, (n - 1) / (t - 1))          # the source must be long enough
        need = rate * (t - 1) + 1
        start = min(float(i), n - need)              # stay inside the video
        a = int(start)
        src = slice(a, min(n, a + int(np.ceil(need)) + 2))
        return time_stretch(d["frames"][src], d["ppg"][src], rate, start - a, t)
