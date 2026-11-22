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

    def process(self, bgr):
        """-> (478, 3) normalised landmarks of the first face, or None."""
        import mediapipe as mp
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts = max(int((time.monotonic() - self._t0) * 1000), self._last_ts + 1)   # must strictly increase
        self._last_ts = ts
        res = self.landmarker.detect_for_video(img, ts)
        if not res.face_landmarks:
            return None
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


def draw_rois(frame, polygons):
    overlay = frame.copy()
    for name, poly in polygons.items():
        cv2.fillConvexPoly(overlay, poly, ROI_COLORS[name])
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)
    for name, poly in polygons.items():
        cv2.polylines(frame, [poly], True, ROI_COLORS[name], 1, cv2.LINE_AA)
    return frame
