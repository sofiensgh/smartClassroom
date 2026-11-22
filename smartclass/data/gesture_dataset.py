"""Step 2 - our own gesture dataset on disk.

Layout:  data/gestures/<person>/<gesture>/<nnnn>.npz

Each .npz holds ONE sample of SEQ_LEN frames:
    landmarks  (T, 21, 3) float32  raw MediaPipe coords (x, y in [0,1] of the image, z relative depth);
                                   zeros where no hand was found. Raw = keeps the hand MOTION (swipes).
    present    (T,) bool           hand found in that frame
    handedness (T,) str            "Left" / "Right" / ""
    crops_jpg  (N,) uint8          optional: the T hand crops (CROP_SIZE x CROP_SIZE, BGR) as concatenated JPEGs
    crops_off  (T+1,) int64        byte offsets of each JPEG in crops_jpg (empty JPEG = no hand)
    gesture, person (str), fps (float), mirror (bool), created (unix time)

Only plain arrays are stored, so files load with allow_pickle=False (safe).
The person id is a pseudonym (p01, p02...): never a real name.
"""
import os
import re
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from ..config import GESTURE_DIR, GESTURES

_PERSON_RE = re.compile(r"[^A-Za-z0-9_-]")


def clean_person_id(text):
    return _PERSON_RE.sub("", text.strip())[:20]


def sample_dir(person, gesture, root=GESTURE_DIR):
    return Path(root) / person / gesture


def _next_index(folder):
    nums = [int(p.stem) for p in folder.glob("*.npz") if p.stem.isdigit()]
    return max(nums, default=-1) + 1


def save_sample(person, gesture, landmarks, present, handedness, crops=None,
                fps=0.0, mirror=True, root=GESTURE_DIR):
    """Write one sample and return its path. `crops`: list of T BGR images or None per frame."""
    if gesture not in GESTURES:
        raise ValueError(f"unknown gesture {gesture!r}")
    folder = sample_dir(person, gesture, root)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {
        "landmarks": np.asarray(landmarks, np.float32),
        "present": np.asarray(present, bool),
        "handedness": np.asarray(handedness, dtype="<U5"),
        "gesture": np.array(gesture), "person": np.array(person),
        "fps": np.float32(fps), "mirror": np.bool_(mirror), "created": np.float64(time.time()),
    }
    if crops is not None:
        blobs = [cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].ravel() if c is not None
                 else np.zeros(0, np.uint8) for c in crops]
        arrays["crops_jpg"] = np.concatenate(blobs) if blobs else np.zeros(0, np.uint8)
        arrays["crops_off"] = np.concatenate([[0], np.cumsum([len(b) for b in blobs])]).astype(np.int64)
    path = folder / f"{_next_index(folder):04d}.npz"
    # Atomic + verified write: a half-written or zero-filled file (seen once in the OneDrive folder)
    # must never appear under a real sample name.
    tmp = path.with_name(path.stem + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        if not zipfile.is_zipfile(tmp):
            raise OSError(f"verification failed for {tmp.name}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def is_valid_sample(path):
    try:
        return zipfile.is_zipfile(path)
    except OSError:
        return False


def load_sample(path, with_crops=False, crop_size=96):
    """Read a sample as a dict. Crops come back as (T, S, S, 3) RGB uint8 (black where no hand)."""
    with np.load(path, allow_pickle=False) as z:
        s = {k: z[k] for k in z.files if not k.startswith("crops")}
        if with_crops and "crops_jpg" in z.files:
            buf, off = z["crops_jpg"], z["crops_off"]
            crops = np.zeros((len(off) - 1, crop_size, crop_size, 3), np.uint8)
            for i in range(len(off) - 1):
                if off[i + 1] > off[i]:
                    img = cv2.imdecode(buf[off[i]:off[i + 1]], cv2.IMREAD_COLOR)
                    crops[i] = cv2.cvtColor(cv2.resize(img, (crop_size, crop_size)), cv2.COLOR_BGR2RGB)
            s["crops"] = crops
    s["gesture"], s["person"] = str(s["gesture"]), str(s["person"])
    return s


def count_samples(root=GESTURE_DIR):
    """{person: {gesture: n}} and the total size in bytes."""
    counts, size = {}, 0
    root = Path(root)
    if not root.exists():
        return counts, size
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        per = {}
        for g in GESTURES:
            files = list((person_dir / g).glob("*.npz"))
            per[g] = len(files)
            size += sum(f.stat().st_size for f in files)
        if any(per.values()):
            counts[person_dir.name] = per
    return counts, size


def list_samples(root=GESTURE_DIR):
    """[(path, person, gesture)] for every sample: used by the training code (Step 3)."""
    root = Path(root)
    return [(p, p.parent.parent.name, p.parent.name)
            for p in sorted(root.glob("*/*/*.npz")) if p.parent.name in GESTURES]
