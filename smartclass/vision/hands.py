"""Step 1 - hand tracking with MediaPipe Hand Landmarker (21 landmarks per hand).

MediaPipe is only a *tool* here: it turns pixels into hand keypoints. The gesture
recognition models (MLP / LSTM / CNN) are trained by us in Steps 2-3.
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import HAND_MODEL_PATH, HAND_MODEL_URL
from .assets import ensure_file

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]  # joint below each tip (IP for the thumb)
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


@dataclass
class Hand:
    landmarks: np.ndarray   # (21, 3) normalised x, y in [0,1] and relative depth z
    handedness: str         # "Left" / "Right" (as seen by the person, image is mirrored)
    score: float

    def pixels(self, w, h):
        return [(int(x), int(y)) for x, y in self.landmarks[:, :2] * [w, h]]  # plain ints for cv2

    def fingers_up(self):
        """Orientation-independent: a finger is extended if its tip is farther from the wrist than its joint."""
        lm = self.landmarks[:, :2]
        wrist = lm[0]
        up = []
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
            ref = lm[17] if tip == 4 else wrist  # thumb: compare to the pinky base
            up.append(np.linalg.norm(lm[tip] - ref) > np.linalg.norm(lm[pip] - ref) * 1.1)
        return up

    def features(self):
        """Hand-shape vector (63,): see features_from_landmarks."""
        return features_from_landmarks(self.landmarks)


def features_from_landmarks(lms):
    """(..., 21, 3) landmarks -> (..., 63) wrist-centred, scale-normalised hand SHAPE.

    This removes the hand position, so it does not contain the motion. Swipes need
    the wrist trajectory too, which is why the dataset also stores the raw landmarks.
    All-zero frames (no hand) stay zero.
    """
    lms = np.asarray(lms, dtype=np.float32)
    pts = lms - lms[..., :1, :]
    scale = np.linalg.norm(pts[..., :2], axis=-1).max(axis=-1)
    scale = np.where(scale > 0, scale, 1.0)[..., None, None]
    return (pts / scale).reshape(*lms.shape[:-2], 63).astype(np.float32)


def crop_hand(image, landmarks, size=96, margin=0.25):
    """Square crop around the hand (padded at the image border), resized to size x size."""
    h, w = image.shape[:2]
    xy = landmarks[:, :2] * [w, h]
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0, 32) * (0.5 + margin)
    xa, ya, xb, yb = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    pad = min(max(0, -xa, -ya, xb - w, yb - h), max(w, h))
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        xa, ya, xb, yb = xa + pad, ya + pad, xb + pad, yb + pad
    crop = image[max(ya, 0):yb, max(xa, 0):xb]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


class HandTracker:
    def __init__(self, num_hands=2, status=None):
        from mediapipe.tasks.python import BaseOptions, vision
        model = ensure_file(HAND_MODEL_URL, HAND_MODEL_PATH, status)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=0.6, min_tracking_confidence=0.5)
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self._t0 = time.monotonic()
        self._last_ts = -1

    def process(self, bgr):
        import mediapipe as mp
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts = max(int((time.monotonic() - self._t0) * 1000), self._last_ts + 1)  # must increase
        self._last_ts = ts
        res = self.landmarker.detect_for_video(img, ts)
        hands = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            arr = np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float32)
            hands.append(Hand(arr, hd[0].category_name, float(hd[0].score)))
        return hands

    def close(self):
        self.landmarker.close()


COLORS = {"Left": (255, 170, 60), "Right": (80, 200, 255)}


def draw_hands(frame, hands):
    h, w = frame.shape[:2]
    for hand in hands:
        pts = hand.pixels(w, h)
        col = COLORS.get(hand.handedness, (200, 200, 200))
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], col, 2, cv2.LINE_AA)
        up = hand.fingers_up()
        for i, p in enumerate(pts):
            is_tip = i in FINGER_TIPS
            lit = is_tip and up[FINGER_TIPS.index(i)]
            cv2.circle(frame, p, 7 if is_tip else 4, (60, 255, 120) if lit else (255, 255, 255), -1, cv2.LINE_AA)
        x, y = pts[0]
        cv2.putText(frame, f"{hand.handedness} {sum(up)}", (x - 30, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return frame
