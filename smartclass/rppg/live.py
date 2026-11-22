"""Step 5 - live heart rate processor (runs in the CameraWorker thread).

frame -> FaceLandmarker -> forehead + cheek ROIs -> mean RGB -> PulseEstimator (POS) every 0.5 s
"""
from ..config import HR_MIN_S
from ..vision.face import FaceTracker, draw_rois, roi_mean_rgb, roi_polygons
from .pos import PulseEstimator

ESTIMATE_EVERY_S = 0.5


class HeartRateProcessor:
    def __init__(self, status=None):
        self.face = FaceTracker(status=status)
        self.est = PulseEstimator()
        self._last_t = -1e9
        self._out = None
        self._seq = 0   # increases with every new estimate (the UI redraws only then)

    def __call__(self, frame, ctx):
        t = ctx["t"]
        h, w = frame.shape[:2]
        lms = self.face.process(frame)
        if lms is None:
            self.est.push(t, None)
        else:
            polys = roi_polygons(lms, w, h)
            self.est.push(t, roi_mean_rgb(frame, polys))   # read the pixels BEFORE drawing on them
            if ctx["draw"]:
                draw_rois(frame, polys)
        if self.est.span < HR_MIN_S:
            self._out = None
        elif t - self._last_t >= ESTIMATE_EVERY_S:
            self._out = self.est.estimate()
            self._last_t = t
            self._seq += 1
        return {"face": lms is not None, "hr": self._out, "hr_seq": self._seq, "hr_progress": min(self.est.span / HR_MIN_S, 1.0)}

    def close(self):
        self.face.close()
