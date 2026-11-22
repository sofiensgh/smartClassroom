"""Pack the code + the gesture dataset into one zip for training on Colab / Kaggle.

    python -m smartclass.training.pack          ->  SmartClassroom_train.zip  (in the project folder)

Upload that zip to Colab (notebooks/train_gestures_colab.ipynb). The notebook gives back
SmartClassroom_results.zip: unzip it INTO the project folder (it contains models/ and output/).
"""
import zipfile
from pathlib import Path

from ..config import GESTURE_DIR, ROOT
from ..data.gesture_dataset import count_samples


def main():
    counts, size = count_samples()
    if not counts:
        raise SystemExit(f"No samples in {GESTURE_DIR}: record gestures in the app first.")
    out = ROOT / "SmartClassroom_train.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT / "smartclass").rglob("*.py"):
            z.write(p, p.relative_to(ROOT))
        for p in GESTURE_DIR.rglob("*.npz"):   # already compressed -> stored as is
            z.write(p, p.relative_to(ROOT), compress_type=zipfile.ZIP_STORED)
            n += 1
    people = ", ".join(f"{p} ({sum(c.values())})" for p, c in counts.items())
    print(f"{out.name}: {n} samples from {len(counts)} people [{people}], {out.stat().st_size / 1e6:.1f} MB")
    print("Next: upload it in notebooks/train_gestures_colab.ipynb")


if __name__ == "__main__":
    main()
