"""Step 7 - the two model inputs of one 20 s window, SHARED by training and live inference.

    face  (200, 52)  MediaPipe blendshape scores (brows, eyes, mouth...) resampled to 10 Hz
    pulse (600,)     POS pulse wave (forehead + cheeks colour) at 30 Hz, band-passed, z-scored

MediaPipe and POS are tools: they turn pixels into numbers. Deciding "stressed or not" is the job of
the deep model (LSTM on the face sequence + 1D CNN on the pulse wave, fused).
"""
import numpy as np

from ..config import STRESS_FACE_FS, STRESS_PULSE_FS, STRESS_WIN_S
from ..rppg.pos import bandpass, pos

N_BLEND = 52
MIN_FACE_RATIO = 0.7


def _interp(t, values, grid):
    return np.stack([np.interp(grid, t, values[:, c]) for c in range(values.shape[1])], 1)


def window_inputs(t, blend, rgb, t_end, win_s=STRESS_WIN_S):
    """t (N,), blend (N, 52) and rgb (N, 3) with NaN rows where no face -> (face, pulse) or None."""
    t = np.asarray(t, np.float64)
    sel = (t >= t_end - win_s) & (t <= t_end)
    if sel.sum() < 10:
        return None
    ts, b, c = t[sel], np.asarray(blend)[sel], np.asarray(rgb)[sel]
    ok = ~np.isnan(b).any(1) & ~np.isnan(c).any(1)
    if ok.mean() < MIN_FACE_RATIO or ts[-1] - ts[0] < 0.9 * win_s:
        return None
    ts, b, c = ts[ok], b[ok], c[ok]
    t0 = t_end - win_s
    face = _interp(ts, b, t0 + np.arange(int(win_s * STRESS_FACE_FS)) / STRESS_FACE_FS).astype(np.float32)
    rgb_u = _interp(ts, c, t0 + np.arange(int(win_s * STRESS_PULSE_FS)) / STRESS_PULSE_FS)
    wave = bandpass(pos(rgb_u, STRESS_PULSE_FS), STRESS_PULSE_FS)
    wave = ((wave - wave.mean()) / (wave.std() + 1e-9)).astype(np.float32)
    return face, wave


def session_windows(d, step_s, win_s=STRESS_WIN_S):
    """All windows of one recorded session -> (faces (W,200,52), pulses (W,600), t_ends)."""
    t = d["t"]
    faces, pulses, ends = [], [], []
    for t_end in np.arange(t[0] + win_s, t[-1] + 1e-6, step_s):
        w = window_inputs(t, d["blend"], d["rgb"], t_end, win_s)
        if w is not None:
            faces.append(w[0])
            pulses.append(w[1])
            ends.append(t_end - t[0])
    if not faces:
        return np.zeros((0, int(win_s * STRESS_FACE_FS), N_BLEND), np.float32), np.zeros((0, int(win_s * STRESS_PULSE_FS)), np.float32), []
    return np.stack(faces), np.stack(pulses), ends
