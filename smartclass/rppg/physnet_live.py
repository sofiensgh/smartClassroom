"""Step 6b - PhysNet (our trained 3D CNN) live, next to POS.

Camera thread : FaceCropper 72x72 crop (same as the UBFC preprocessing) -> buffer (time, crop)
PhysNet thread: every ~1.5 s, the last 10 s of crops -> resampled to 30 fps (UBFC rate) ->
                PhysNet (overlapping 128-frame chunks, same predict_signal as the test evaluation)
                -> band-pass -> spectrum peak = BPM
The network takes ~1 s on the CPU, so it runs in its own thread: the camera never waits for it.
"""
import threading
from collections import deque

import numpy as np

from ..config import HR_MIN_S, HR_WINDOW_S, MODELS_DIR, PHYS_FS, PHYS_SIZE, PHYS_T
from .pos import bandpass, spectrum_bpm

PHYSNET_PATH = MODELS_DIR / "physnet.pt"
ESTIMATE_EVERY_S = 1.5


def physnet_available():
    return PHYSNET_PATH.exists()


def resample_frames(t, frames, fs):
    """Irregular webcam frames -> uniform fs grid (linear blend of the 2 nearest frames)."""
    tu = np.arange(t[0], t[-1], 1 / fs)
    j = np.clip(np.searchsorted(t, tu, side="right") - 1, 0, len(t) - 2)
    a = ((tu - t[j]) / np.maximum(t[j + 1] - t[j], 1e-6)).clip(0, 1)[:, None, None, None]
    f = frames.astype(np.float32)
    return np.clip(f[j] * (1 - a) + f[j + 1] * a + 0.5, 0, 255).astype(np.uint8)


class PhysNetEstimator:
    def __init__(self, path=PHYSNET_PATH, window_s=HR_WINDOW_S, min_s=max(HR_MIN_S, PHYS_T / PHYS_FS + 0.5),
                 max_gap_s=1.0, threads=2):
        import torch

        from ..models.physnet import PhysNet
        from ..training.rppg_data import predict_signal
        torch.set_num_threads(threads)          # leave CPU cores to the camera + MediaPipe
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.model = PhysNet(ck.get("width", 32))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.size, self.fs = int(ck.get("size", PHYS_SIZE)), float(ck.get("fs", PHYS_FS))
        self.val_mae = ck.get("val_mae")
        self._predict = lambda frames: predict_signal(self.model, frames, torch.device("cpu"))
        self.window_s, self.min_s, self.max_gap_s = window_s, min_s, max_gap_s
        self.t, self.frames = deque(), deque()
        self.history = deque(maxlen=5)
        self._lock = threading.Lock()
        self._job = None                        # latest (t, frames) snapshot waiting for the network
        self._wake = threading.Event()
        self._closed = False
        self.result, self.seq = None, 0
        self._thread = threading.Thread(target=self._loop, name="physnet", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- camera thread
    @property
    def span(self):
        return self.t[-1] - self.t[0] if len(self.t) > 1 else 0.0

    def reset(self):
        self.t.clear()
        self.frames.clear()
        self.history.clear()
        with self._lock:
            self.result = None

    def push(self, t, crop_rgb):
        if self.t and t - self.t[-1] > self.max_gap_s:   # face lost for too long: restart
            self.reset()
        if crop_rgb is None:
            return
        self.t.append(t)
        self.frames.append(crop_rgb)
        while self.t and t - self.t[0] > self.window_s:
            self.t.popleft()
            self.frames.popleft()

    def request(self):
        """Hand the current window to the PhysNet thread (an older waiting one is replaced)."""
        if self.span < self.min_s:
            return
        with self._lock:
            self._job = (np.asarray(self.t), np.stack(self.frames))
        self._wake.set()

    def latest(self):
        with self._lock:
            return self.result, self.seq

    # ---------------------------------------------------------------- PhysNet thread
    def _loop(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._closed:
                return
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            try:
                res = self._estimate(*job)
            except Exception as e:  # noqa: BLE001 - never kill the thread; show the error instead
                res = {"bpm": None, "error": f"{type(e).__name__}: {e}"}
            with self._lock:
                self.result = res
                self.seq += 1

    def _estimate(self, t, frames):
        clip = resample_frames(t, frames, self.fs)
        if len(clip) < PHYS_T:
            return None
        sig = bandpass(self._predict(clip), self.fs)
        bpm, snr, fb, p = spectrum_bpm(sig, self.fs)
        self.history.append(bpm)
        tail = sig[-int(6 * self.fs):]
        return {"bpm": float(np.median(self.history)), "bpm_raw": bpm, "snr": snr,
                "wave": tail / max(np.abs(tail).max(), 1e-12), "spec_bpm": fb, "spec": p}

    def close(self):
        self._closed = True
        self._wake.set()
        self._thread.join(timeout=3)
