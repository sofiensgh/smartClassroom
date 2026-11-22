"""Step 6 - UBFC-rPPG (DATASET_2) -> compact per-subject cache for PhysNet and POS.

    python -m smartclass.rppg.preprocess --ubfc /path/to/UBFC_DATASET/DATASET_2 --out data/ubfc_cache

Expected layout (official DATASET_2): <root>/subject<N>/vid.avi + ground_truth.txt
ground_truth.txt = 3 rows: PPG signal, heart rate (BPM), timestamps (s).

For every frame we keep (videos are several GB; the cache is ~30 MB per subject):
    frames  (T, 72, 72, 3) uint8 RGB   face crop (FaceCropper, same as live)   -> PhysNet input
    rgb     (T, 3) float32             mean skin colour of the POS ROIs         -> POS baseline on the same frames
    ppg     (T,) float32               ground-truth PPG resampled to the frames -> training label / reference HR
    face    (T,) bool                  face found in that frame
"""
import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import PHYS_SIZE, UBFC_CACHE


def find_subjects(root):
    """[(subject_id, video_path, gt_path)] sorted by subject number."""
    out = []
    for d in Path(root).iterdir():
        m = re.fullmatch(r"subject(\d+)", d.name)
        if m and (d / "vid.avi").exists() and (d / "ground_truth.txt").exists():
            out.append((f"subject{int(m.group(1)):02d}", d / "vid.avi", d / "ground_truth.txt"))
    return sorted(out)


def read_ground_truth(path):
    """-> ppg, hr, t (seconds) from a DATASET_2 ground_truth.txt."""
    rows = [np.array(line.split(), dtype=np.float64) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) < 3:
        raise ValueError(f"{path}: expected 3 rows (PPG, HR, time), got {len(rows)}")
    return rows[0], rows[1], rows[2]


def preprocess_video(video_path, gt_path, tracker=None, size=PHYS_SIZE, log=print):
    from ..vision.face import FaceCropper, FaceTracker, roi_mean_rgb, roi_polygons
    tracker = tracker or FaceTracker()
    cropper = FaceCropper(size=size)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, rgb, face = [], [], []
    last_rgb = np.full(3, np.nan)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        lms = tracker.process(bgr)
        crop = cropper(bgr, lms)
        if lms is not None:
            v = roi_mean_rgb(bgr, roi_polygons(lms, bgr.shape[1], bgr.shape[0]))
            last_rgb = v if v is not None else last_rgb
        frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop is not None
                      else np.zeros((size, size, 3), np.uint8))
        rgb.append(last_rgb)
        face.append(lms is not None)
    cap.release()
    if not frames:
        raise ValueError(f"cannot read {video_path}")
    frames, rgb, face = np.stack(frames), np.array(rgb, np.float32), np.array(face)
    # frames before the first face: copy the first good value (keeps the arrays aligned)
    first = int(np.argmax(face)) if face.any() else 0
    frames[:first] = frames[first]
    rgb[:first] = rgb[first]
    ppg, hr, t_gt = read_ground_truth(gt_path)
    t_frames = np.arange(len(frames)) / fps
    if len(t_gt) != len(ppg) or t_gt.max() <= 0:    # some files have no usable timestamps
        t_gt = np.arange(len(ppg)) / fps
    ppg_f = np.interp(t_frames, t_gt - t_gt[0], ppg).astype(np.float32)
    return {"frames": frames, "rgb": rgb, "ppg": ppg_f, "face": face, "fps": np.float32(fps),
            "gt_hr_mean": np.float32(np.mean(hr))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ubfc", type=Path, required=True, help="folder containing subject*/ (DATASET_2)")
    ap.add_argument("--out", type=Path, default=UBFC_CACHE)
    ap.add_argument("--limit", type=int, default=0, help="only the first N subjects (quick test)")
    a = ap.parse_args()
    subjects = find_subjects(a.ubfc)
    if a.limit:
        subjects = subjects[:a.limit]
    if not subjects:
        raise SystemExit(f"No subject*/vid.avi + ground_truth.txt found in {a.ubfc}")
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(subjects)} subjects -> {a.out}")
    from ..vision.face import FaceTracker
    for i, (sid, vid, gt) in enumerate(subjects, 1):
        out = a.out / f"{sid}.npz"
        if out.exists():
            print(f"[{i}/{len(subjects)}] {sid}: cached")
            continue
        t0 = time.time()
        d = preprocess_video(vid, gt, tracker=FaceTracker())   # new tracker: timestamps restart per video
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, **d)
        tmp.replace(out)
        print(f"[{i}/{len(subjects)}] {sid}: {len(d['frames'])} frames, face {d['face'].mean():.0%}, "
              f"mean HR {d['gt_hr_mean']:.0f} BPM, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
