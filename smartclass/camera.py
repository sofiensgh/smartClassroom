"""Background camera thread: grabs frames, runs the vision processors, sends results to the UI."""
import sys
import time

import cv2
from PySide6.QtCore import QThread, Signal

from .config import CAMERA_HEIGHT, CAMERA_WIDTH, CROP_SIZE
from .vision.hands import crop_hand, draw_hands


class CameraWorker(QThread):
    frame_ready = Signal(object, dict)  # RGB numpy image, info dict
    status = Signal(str)
    error = Signal(str)

    def __init__(self, index=0, mirror=True, track_hands=True, draw=True, crops=False, lock_exposure=False,
                 parent=None):
        super().__init__(parent)
        self.index, self.mirror, self.track_hands, self.draw = index, mirror, track_hands, draw
        self.crops = crops  # also send CROP_SIZE hand crops (BGR), one per hand
        self.lock_exposure = lock_exposure
        # Set once by stop() and never reset by run(): a stop requested while the camera
        # is still opening (or the model downloading) must not be lost.
        self._stop_requested = False
        # Optional per-frame processor (e.g. the gesture model), BUILT in this thread by a factory
        # so heavy imports (torch) never block the UI. Replaceable while running.
        self._processor_factory = None
        self._processor_changed = False
        self.processor = None

    def set_processor_factory(self, factory):
        """factory() -> callable(frame_bgr, ctx) -> dict merged into the frame info. None = no processor.

        ctx = {"hands", "mirror", "draw", "t"}. The processor may draw on frame_bgr (it is shown after),
        but must read the pixels it needs BEFORE drawing."""
        self._processor_factory = factory
        self._processor_changed = True

    def stop(self, timeout_ms=5000):
        """Ask the thread to finish and wait for it. Returns True if it has finished."""
        self._stop_requested = True
        return self.wait(timeout_ms)

    def run(self):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY  # DirectShow opens faster on Windows
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            self.error.emit(f"Cannot open camera {self.index} (is another app using it?)")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        tracker = None
        if self.track_hands and not self._stop_requested:
            try:
                from .vision.hands import HandTracker  # imports mediapipe (slow), so only here
                tracker = HandTracker(status=self.status.emit)
                self.status.emit("Hand tracking ready")
            except Exception as e:  # noqa: BLE001 - shown to the user
                self.error.emit(f"Hand tracker failed: {e}")

        locked = self.lock_exposure and not self._stop_requested and self._lock_exposure(cap)

        fps, last = 0.0, time.monotonic()
        while not self._stop_requested:
            ok, frame = cap.read()
            if not ok:
                self.error.emit("Camera stopped sending frames")
                break
            if self.mirror:
                frame = cv2.flip(frame, 1)
            hands = tracker.process(frame) if tracker else []
            info = {"hands": hands, "mirror": self.mirror}
            if self._processor_changed:
                self._build_processor()
            if self.processor is not None:
                try:
                    ctx = {"hands": hands, "mirror": self.mirror, "draw": self.draw, "t": time.monotonic()}
                    info.update(self.processor(frame, ctx))
                except Exception as e:  # noqa: BLE001 - shown to the user, camera keeps running
                    self.error.emit(f"Processing error: {e}")
                    self.processor = None
            # crops are cut from the clean frame, before the landmarks are drawn on it
            crops = [crop_hand(frame, h.landmarks, CROP_SIZE) for h in hands] if self.crops else None
            if self.draw and hands:
                draw_hands(frame, hands)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            info.update(fps=fps, crops=crops)
            self.frame_ready.emit(rgb, info)
        if locked:   # the driver keeps the setting: give the camera back in auto mode
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        cap.release()
        if tracker:
            tracker.close()
        self._close_processor()

    def _lock_exposure(self, cap, exposures=(-8, -7, -6, -5)):
        """Heart rate: the camera's AUTO exposure / white balance change the image brightness in steps
        (measured: skin green 150 -> 124 -> 88 in 40 s), which POS and PhysNet see as a fake pulse.
        Let auto exposure settle, then fix the exposure giving the same brightness (>= -5 keeps 30 fps)."""
        self.status.emit("Camera: measuring the light (2 s)...")
        vals = []
        for _ in range(40):
            ok, f = cap.read()
            if not ok or self._stop_requested:
                return False
            vals.append(f.mean())
        target = sum(vals[-10:]) / 10
        if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25):
            self.status.emit("Camera: exposure cannot be fixed on this webcam (auto exposure kept)")
            return False
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        best = None
        for e in exposures:
            cap.set(cv2.CAP_PROP_EXPOSURE, e)
            for _ in range(10):          # the new exposure needs several frames to settle
                ok, f = cap.read()
            if not ok:
                break
            b = f.mean()
            if best is None or abs(b - target) < abs(best[1] - target):
                best = (e, b)
        if best is None:
            return True
        cap.set(cv2.CAP_PROP_EXPOSURE, best[0])
        self.status.emit(f"Camera: exposure fixed at {best[0]} (brightness {best[1]:.0f}, auto was {target:.0f})")
        return True

    def _close_processor(self):
        close = getattr(self.processor, "close", None)
        if close:
            try:
                close()
            except Exception:  # noqa: BLE001 - best effort on shutdown
                pass
        self.processor = None

    def _build_processor(self):
        self._processor_changed = False
        self._close_processor()
        if self._processor_factory is None:
            return
        try:
            self.processor = self._processor_factory()
            self.status.emit("Processing ready")
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.error.emit(f"Cannot load the gesture model: {e}")
