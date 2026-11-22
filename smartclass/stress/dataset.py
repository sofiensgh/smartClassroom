"""Step 7 - pilot stress dataset: data/stress/<person>/<rest|stress>_<nn>.npz

Each file = one session (default 2 min), NUMBERS ONLY (ethics: no video is stored):
    t (N,) seconds, blend (N, 52) blendshape scores, rgb (N, 3) mean skin colour (NaN rows = no face),
    label 'rest' | 'stress', person, task_score (arithmetic answers right / asked, stress sessions only)
Protocol (like UBFC-Phys T1 vs T3): REST = sit, breathe slowly following the app; STRESS = timed mental
arithmetic in the app (3 choices, 4 s per question, beep when wrong or too slow).
"""
import os
import re

import numpy as np

from ..config import STRESS_DIR, STRESS_LABELS


def next_path(person, label, root=STRESS_DIR):
    d = root / person
    d.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for p in d.glob(f"{label}_*.npz") if (m := re.fullmatch(rf"{label}_(\d+)", p.stem))]
    return d / f"{label}_{max(nums, default=0) + 1:02d}.npz"


def save_session(person, label, t, blend, rgb, task_score=None, root=STRESS_DIR):
    path = next_path(person, label, root)
    tmp = path.with_name(path.stem + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, t=np.asarray(t, np.float64), blend=np.asarray(blend, np.float32),
                            rgb=np.asarray(rgb, np.float32), label=label, person=person,
                            task_score=np.float32(np.nan if task_score is None else task_score))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    return path


def list_sessions(root=STRESS_DIR):
    """[(path, person, label)] sorted."""
    out = []
    for p in sorted(root.glob("*/*.npz")):
        m = re.fullmatch(r"(rest|stress)_(\d+)", p.stem)
        if m and m.group(1) in STRESS_LABELS:
            out.append((p, p.parent.name, m.group(1)))
    return out


def load_session(path):
    with np.load(path) as z:
        return {k: z[k] for k in ("t", "blend", "rgb")} | {"label": str(z["label"]), "person": str(z["person"])}


def count_sessions(root=STRESS_DIR):
    counts = {}
    for _, person, label in list_sessions(root):
        counts.setdefault(person, {k: 0 for k in STRESS_LABELS})[label] += 1
    return counts
