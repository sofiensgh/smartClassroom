"""Landmark sequence -> model input. Shared by training (Step 3) and live inference (Step 4),
so the model always sees exactly the same features.

Per frame (68 values):
    63  hand SHAPE: wrist-centred, scale-normalised landmarks           (what the hand looks like)
     2  wrist POSITION relative to the mean wrist position of the window (where it is)
     2  wrist VELOCITY (frame-to-frame)                                  (how it moves -> swipes)
     1  PRESENT flag (1 = a hand was found in this frame)
Frames without a hand are all zeros. With motion=False only shape + present (64 values): ablation.
"""
import numpy as np

from ..vision.hands import features_from_landmarks

POS_SCALE = 5.0    # wrist offsets are ~0.0-0.3 of the image  -> ~0-1.5
VEL_SCALE = 20.0   # frame-to-frame motion is ~0.0-0.05       -> ~0-1

MIRROR_SWAP = {"swipe_left": "swipe_right", "swipe_right": "swipe_left"}  # label after a left/right flip


def feature_dim(motion=True):
    return 68 if motion else 64


def fix_length(landmarks, present, seq_len):
    """Crop or zero-pad (at the start) to exactly seq_len frames, keeping the most recent ones."""
    t = len(present)
    if t >= seq_len:
        return landmarks[-seq_len:], present[-seq_len:]
    pad = seq_len - t
    return (np.concatenate([np.zeros((pad, 21, 3), np.float32), landmarks]),
            np.concatenate([np.zeros(pad, bool), present]))


def mirror_x(landmarks, present):
    """Flip left/right (x -> 1-x) on the frames that contain a hand."""
    out = landmarks.copy()
    out[present, :, 0] = 1.0 - out[present, :, 0]
    return out


def sequence_features(landmarks, present, motion=True):
    """(T, 21, 3) raw landmarks + (T,) present -> (T, 68) float32 (or (T, 64) without motion)."""
    landmarks = np.asarray(landmarks, np.float32)
    present = np.asarray(present, bool)
    shape = features_from_landmarks(landmarks) * present[:, None]
    flag = present[:, None].astype(np.float32)
    if not motion:
        return np.concatenate([shape, flag], 1).astype(np.float32)
    wrist = landmarks[:, 0, :2]
    pos = np.zeros_like(wrist)
    vel = np.zeros_like(wrist)
    if present.any():
        pos[present] = (wrist[present] - wrist[present].mean(0)) * POS_SCALE
        both = present[1:] & present[:-1]
        vel[1:][both] = (wrist[1:] - wrist[:-1])[both] * VEL_SCALE
    return np.concatenate([shape, pos, vel, flag], 1).astype(np.float32)
