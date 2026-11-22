"""Steps 5 + 6b - live heart rate processor (runs in the CameraWorker thread).

frame -> FaceLandmarker -> forehead + cheek ROIs -> mean RGB -> PulseEstimator (POS) every 0.5 s
                        -> 72x72 face crop (FaceCropper)  -> PhysNetEstimator (own thread) every 1.5 s
Both methods see the SAME frames, so the page can compare them side by side.
"""
import cv2

from ..config import HR_MIN_S, PHYS_SIZE
from ..vision.face import FaceCropper, FaceTracker, draw_rois, roi_mean_rgb, roi_polygons
from .pos import PulseEstimator

ESTIMATE_EVERY_S = 0.5


class HeartRateProcessor:
    def __init__(self, status=None):
        self.face = FaceTracker(status=status)
        self.est = PulseEstimator()
        self._last_t = -1e9
        self._out = None
        self._seq = 0   # increases with every new estimate (the UI redraws only then)
        self.phys, self.phys_error = None, None
        self._phys_t = -1e9
        try:
            from .physnet_live import ESTIMATE_EVERY_S as phys_every, PhysNetEstimator, physnet_available
            if physnet_available():
                self.phys = PhysNetEstimator()
                self.cropper = FaceCropper(size=self.phys.size)
                self._phys_every = phys_every
            else:
                self.phys_error = "models/physnet.pt not found (train it: Step 6)"
        except Exception as e:  # noqa: BLE001 - POS must keep working without PhysNet
            self.phys_error = f"PhysNet not loaded: {type(e).__name__}: {e}"
        if self.phys is None:
            self.cropper = FaceCropper(size=PHYS_SIZE)

    def __call__(self, frame, ctx):
        t = ctx["t"]
        h, w = frame.shape[:2]
        lms = self.face.process(frame)
        # read the pixels BEFORE drawing on them
        if self.phys is not None:
            crop = self.cropper(frame, lms) if lms is not None else None
            self.phys.push(t, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop is not None else None)
        if lms is None:
            self.est.push(t, None)
        else:
            polys = roi_polygons(lms, w, h)
            self.est.push(t, roi_mean_rgb(frame, polys))
            if ctx["draw"]:
                draw_rois(frame, polys)
        if self.est.span < HR_MIN_S:
            self._out = None
        elif t - self._last_t >= ESTIMATE_EVERY_S:
            self._out = self.est.estimate()
            self._last_t = t
            self._seq += 1
        out = {"face": lms is not None, "hr": self._out, "hr_seq": self._seq,
               "hr_progress": min(self.est.span / HR_MIN_S, 1.0), "phys_error": self.phys_error}
        if self.phys is not None:
            if t - self._phys_t >= self._phys_every:
                self.phys.request()
                self._phys_t = t
            res, seq = self.phys.latest()
            out.update(phys=res, phys_seq=seq, phys_progress=min(self.phys.span / self.phys.min_s, 1.0))
        return out

    def close(self):
        if self.phys is not None:
            self.phys.close()
        self.face.close()
