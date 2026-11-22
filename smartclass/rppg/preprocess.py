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
import json
import re
import shutil
import subprocess
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


def read_video(path):
    """-> (fps, n_frames or 0, iterator of BGR frames).

    Uses the ffmpeg command line when it exists (Colab): OpenCV's decoder crashed silently on the
    large UBFC AVI files there. Falls back to OpenCV (e.g. on Windows without ffmpeg)."""
    path = str(path)
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        info = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate,nb_frames", "-of", "json", path],
            capture_output=True, text=True, check=True).stdout)["streams"][0]
        w, h = int(info["width"]), int(info["height"])
        num, den = (int(v) for v in info["r_frame_rate"].split("/"))
        n = int(info["nb_frames"]) if str(info.get("nb_frames", "")).isdigit() else 0

        def frames():
            proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                                    stdout=subprocess.PIPE)
            size = w * h * 3
            try:
                while True:
                    buf = proc.stdout.read(size)
                    if len(buf) < size:
                        break
                    yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            finally:
                proc.stdout.close()
                proc.kill()
                proc.wait()
        return (num / den if den else 30.0), n, frames()

    cap = cv2.VideoCapture(path)
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def frames():
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield bgr
        finally:
            cap.release()
    return fps, n, frames()


def preprocess_video(video_path, gt_path, tracker=None, size=PHYS_SIZE, log=print, detect_every=2):
    from ..vision.face import FaceCropper, FaceTracker, roi_mean_rgb, roi_polygons
    tracker = tracker or FaceTracker()
    cropper = FaceCropper(size=size)
    fps, n_total, video = read_video(video_path)
    frames, rgb, face = [], [], []
    last_rgb = np.full(3, np.nan)
    lms, t0 = None, time.time()
    for bgr in video:
        i = len(frames)
        # landmarks on every 2nd frame (2x faster; the face barely moves in 1/30 s),
        # but crop + skin colour on EVERY frame
        if i % detect_every == 0:
            lms = tracker.process(bgr)
        if log and i and i % 300 == 0:
            log(f"    frame {i}/{n_total or '?'}  ({time.time() - t0:.0f}s)")
        crop = cropper(bgr, lms)
        if lms is not None:
            v = roi_mean_rgb(bgr, roi_polygons(lms, bgr.shape[1], bgr.shape[0]))
            last_rgb = v if v is not None else last_rgb
        frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop is not None
                      else np.zeros((size, size, 3), np.uint8))
        rgb.append(last_rgb)
        face.append(lms is not None)
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
    log = lambda *m: print(*m, flush=True)   # noqa: E731 - visible progress in Colab
    subjects = find_subjects(a.ubfc)
    if a.limit:
        subjects = subjects[:a.limit]
    if not subjects:
        raise SystemExit(f"No subject*/vid.avi + ground_truth.txt found in {a.ubfc}")
    a.out.mkdir(parents=True, exist_ok=True)
    log(f"{len(subjects)} subjects -> {a.out}")
    failed = []
    from ..vision.face import FaceTracker
    for i, (sid, vid, gt) in enumerate(subjects, 1):
        out = a.out / f"{sid}.npz"
        if out.exists():
            log(f"[{i}/{len(subjects)}] {sid}: cached")
            continue
        t0 = time.time()
        log(f"[{i}/{len(subjects)}] {sid}: reading {vid.stat().st_size / 1e9:.1f} GB video ...")
        try:
            tracker = FaceTracker()   # new tracker per video: its timestamps restart
            d = preprocess_video(vid, gt, tracker=tracker, log=log)
            tracker.close()
            if d["face"].mean() < 0.5:
                raise ValueError(f"face found in only {d['face'].mean():.0%} of the frames")
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez_compressed(tmp, **d)
            tmp.replace(out)
        except Exception as e:  # noqa: BLE001 - one bad video must not stop the other 41
            failed.append(sid)
            log(f"[{i}/{len(subjects)}] {sid}: FAILED ({type(e).__name__}: {e})")
            continue
        log(f"[{i}/{len(subjects)}] {sid}: {len(d['frames'])} frames, face {d['face'].mean():.0%}, "
            f"mean HR {d['gt_hr_mean']:.0f} BPM, {time.time() - t0:.0f}s")
    log(f"done: {len(subjects) - len(failed)} ok" + (f", failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
