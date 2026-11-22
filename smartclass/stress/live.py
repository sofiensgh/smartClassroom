"""Step 7 - stress processor (runs in the CameraWorker thread).

Every frame: FaceLandmarker (+ 52 blendshapes) -> blendshapes + mean skin colour of the POS regions.
  - recording: the UI asks (StressControl) for a 2-min session -> saved as numbers only (no video)
  - live: every 2 s, the last 20 s -> window_inputs -> face / pulse / fusion models -> P(stress)
"""
import threading
from collections import deque

import numpy as np

from ..config import MODELS_DIR, STRESS_STEP_S, STRESS_WIN_S
from ..vision.face import FaceTracker, draw_rois, roi_mean_rgb, roi_polygons
from .features import N_BLEND, window_inputs

MODES = ("fusion", "face", "pulse")


def model_path(mode):
    return MODELS_DIR / f"stress_{mode}.pt"


class StressControl:
    """Shared between the UI thread (requests) and the camera thread (recording)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.request = None     # (person, label, duration_s)
        self.cancel = False
        self.task_score = None  # set by the UI at the end of a stress session

    def start(self, person, label, duration):
        with self.lock:
            self.request, self.cancel = (person, label, duration), False

    def stop(self):
        with self.lock:
            self.cancel = True


class StressProcessor:
    def __init__(self, control, status=None):
        self.control = control
        self.face = FaceTracker(status=status, blendshapes=True)
        self.buf = deque()            # (t, blend, rgb) of the last WIN + 5 s
        self.rec = None
        self.models = {}
        self.error = None
        try:
            import torch

            from ..models.stress_models import StressNet
            torch.set_num_threads(2)
            for mode in MODES:
                p = model_path(mode)
                if p.exists():
                    ck = torch.load(p, map_location="cpu", weights_only=True)
                    m = StressNet(mode)
                    m.load_state_dict(ck["state_dict"])
                    m.eval()
                    self.models[mode] = (m, ck["mean"].numpy(), ck["std"].numpy())
        except Exception as e:  # noqa: BLE001 - recording must work without a model
            self.error = f"stress model not loaded: {type(e).__name__}: {e}"
        self._last_est = -1e9
        self.ema = None
        self.out = None
        self.seq = 0

    def __call__(self, frame, ctx):
        t = ctx["t"]
        h, w = frame.shape[:2]
        lms = self.face.process(frame)
        blend = rgb = None
        if lms is not None and self.face.blendshapes is not None:
            polys = roi_polygons(lms, w, h)
            rgb = roi_mean_rgb(frame, polys)     # read pixels BEFORE drawing
            blend = self.face.blendshapes
            if ctx["draw"]:
                draw_rois(frame, polys)
        nan_b, nan_c = np.full(N_BLEND, np.nan, np.float32), np.full(3, np.nan, np.float32)
        b = blend if blend is not None and len(blend) == N_BLEND else nan_b
        c = rgb if rgb is not None else nan_c
        self.buf.append((t, b, c))
        while self.buf and t - self.buf[0][0] > STRESS_WIN_S + 5:
            self.buf.popleft()
        info = {"face": blend is not None, "stress_error": self.error, "models": sorted(self.models)}
        info.update(self._record(t, b, c))
        if self.models and t - self._last_est >= STRESS_STEP_S:
            self._last_est = t
            self._estimate(t)
        info.update(stress=self.out, stress_seq=self.seq,
                    stress_fill=min((t - self.buf[0][0]) / STRESS_WIN_S, 1.0) if self.buf else 0.0)
        return info

    # ---------------------------------------------------------------- recording
    def _record(self, t, b, c):
        ctl = self.control
        with ctl.lock:
            req, cancel = ctl.request, ctl.cancel
            ctl.request = None
            ctl.cancel = False
        if req:
            person, label, dur = req
            self.rec = {"person": person, "label": label, "dur": dur, "t0": t, "t": [], "b": [], "c": []}
        if self.rec is None:
            return {}
        if cancel:
            self.rec = None
            return {"rec_cancelled": True}
        r = self.rec
        r["t"].append(t - r["t0"])
        r["b"].append(b)
        r["c"].append(c)
        elapsed = t - r["t0"]
        if elapsed < r["dur"]:
            return {"rec_label": r["label"], "rec_elapsed": elapsed, "rec_dur": r["dur"]}
        self.rec = None
        face_ratio = float(np.mean([not np.isnan(x).any() for x in r["b"]]))
        if face_ratio < 0.7:
            return {"rec_failed": f"face visible only {face_ratio:.0%} of the time: session not saved"}
        from .dataset import save_session
        path = save_session(r["person"], r["label"], r["t"], r["b"], r["c"], ctl.task_score)
        return {"rec_saved": str(path), "rec_face": face_ratio}

    # ---------------------------------------------------------------- inference
    def _estimate(self, t):
        import torch
        arr_t = np.array([x[0] for x in self.buf])
        w = window_inputs(arr_t, np.stack([x[1] for x in self.buf]), np.stack([x[2] for x in self.buf]), t)
        if w is None:
            self.out = None
            return
        face, pulse = w
        probs = {}
        with torch.no_grad():
            for mode, (m, mean, std) in self.models.items():
                f = torch.from_numpy(((face - mean) / std)[None].astype(np.float32))
                p = torch.from_numpy(pulse[None])
                probs[mode] = float(torch.sigmoid(m(f, p))[0])
        main = probs.get("fusion", next(iter(probs.values())))
        self.ema = main if self.ema is None else 0.7 * self.ema + 0.3 * main   # ~6 s smoothing
        self.out = {"score": 100 * self.ema, "probs": probs}
        self.seq += 1

    def close(self):
        self.face.close()
