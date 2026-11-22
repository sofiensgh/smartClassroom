"""Step 5 - face landmarks (MediaPipe FaceLandmarker, 478 points) and skin regions for rPPG.

MediaPipe is only a tool here: it tells us WHERE the forehead and cheeks are.
The pulse itself comes from the colour of those regions (POS now, PhysNet in Step 6).
"""
import time

import cv2
import numpy as np

from ..config import FACE_MODEL_PATH, FACE_MODEL_URL
from .assets import ensure_file

# Face-mesh indices of three skin regions with few hair/eye/mouth pixels (convex hull of each set).
ROIS = {
    "forehead": [67, 109, 10, 338, 297, 299, 337, 151, 108, 69],
    "left_cheek": [116, 117, 118, 119, 100, 142, 203, 206, 216, 192, 213, 147, 123],
    "right_cheek": [345, 346, 347, 348, 329, 371, 423, 426, 436, 416, 433, 376, 352],
}
ROI_COLORS = {"forehead": (80, 200, 255), "left_cheek": (120, 230, 140), "right_cheek": (120, 230, 140)}


class FaceTracker:
    def __init__(self, status=None, blendshapes=False):
        from mediapipe.tasks.python import BaseOptions, vision
        model = ensure_file(FACE_MODEL_URL, FACE_MODEL_PATH, status)
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
            output_face_blendshapes=blendshapes)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._t0 = time.monotonic()
        self._last_ts = -1
        self.blendshapes = None   # (52,) scores of the last frame when blendshapes=True (Step 7: stress)

    def process(self, bgr):
        """-> (478, 3) normalised landmarks of the first face, or None."""
        import mediapipe as mp
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts = max(int((time.monotonic() - self._t0) * 1000), self._last_ts + 1)   # must strictly increase
        self._last_ts = ts
        res = self.landmarker.detect_for_video(img, ts)
        if not res.face_landmarks:
            self.blendshapes = None
            return None
        if res.face_blendshapes:
            self.blendshapes = np.array([c.score for c in res.face_blendshapes[0]], dtype=np.float32)
        return np.array([(p.x, p.y, p.z) for p in res.face_landmarks[0]], dtype=np.float32)

    def close(self):
        self.landmarker.close()


def roi_polygons(landmarks, w, h):
    """{name: (K, 2) int32 convex polygon in pixels}."""
    pts = (landmarks[:, :2] * [w, h]).astype(np.int32)
    return {name: cv2.convexHull(pts[idx]).reshape(-1, 2) for name, idx in ROIS.items()}


def roi_mean_rgb(bgr, polygons):
    """Mean R, G, B over all ROI pixels (weighted by area), computed only inside each bounding box."""
    h, w = bgr.shape[:2]
    total, count = np.zeros(3), 0
    for poly in polygons.values():
        x, y, pw, ph = cv2.boundingRect(poly)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + pw, w), min(y + ph, h)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, poly - [x0, y0], 1)
        n = int(mask.sum())
        if n < 20:
            continue
        patch = bgr[y0:y1, x0:x1]
        total += np.array(cv2.mean(patch, mask=mask)[:3]) * n
        count += n
    if count == 0:
        return None
    b, g, r = total / count
    return np.array([r, g, b])


class FaceCropper:
    """Square face crop for PhysNet, IDENTICAL in dataset preprocessing (Step 6) and live use.

    The box comes from the landmark bounding box (enlarged), smoothed over time so the crop does
    not jitter (jitter = fake colour changes = noise in the pulse). Without a face, the last box is kept.
    """

    def __init__(self, size=72, scale=1.2, smooth=0.3):
        self.size, self.scale, self.smooth = size, scale, smooth
        self.box = None   # (cx, cy, side) in pixels

    def __call__(self, frame, landmarks):
        h, w = frame.shape[:2]
        if landmarks is not None:
            xy = landmarks[:, :2] * [w, h]
            (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
            new = np.array([(x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * self.scale])
            self.box = new if self.box is None else self.box + (new - self.box) * self.smooth
        if self.box is None:
            return None
        cx, cy, side = self.box
        half = int(side / 2)
        xa, ya = int(cx) - half, int(cy) - half
        pad = max(0, -xa, -ya, xa + 2 * half - w, ya + 2 * half - h)
        img = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else frame
        crop = img[ya + pad:ya + pad + 2 * half, xa + pad:xa + pad + 2 * half]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)


def draw_rois(frame, polygons):
    overlay = frame.copy()
    for name, poly in polygons.items():
        cv2.fillConvexPoly(overlay, poly, ROI_COLORS[name])
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)
    for name, poly in polygons.items():
        cv2.polylines(frame, [poly], True, ROI_COLORS[name], 1, cv2.LINE_AA)
    return frame
