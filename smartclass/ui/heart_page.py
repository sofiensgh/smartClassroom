"""Heart Rate page. Step 5: classic rPPG baseline (POS) from a normal webcam.

Camera thread: FaceLandmarker -> forehead/cheek ROIs -> mean RGB -> POS -> band-pass -> FFT peak (BPM)
UI thread    : shows BPM, pulse wave, spectrum; logs comparisons with a reference device (smartwatch).
"""
import csv
import time

from PySide6.QtWidgets import QGridLayout, QLabel, QProgressBar, QPushButton, QSpinBox, QWidget

from ..config import HR_BAND, HR_DIR, HR_WINDOW_S
from .camera_page import CameraPage
from .plot import SignalPlot
from .widgets import card, label

GOOD_SNR_DB = 0.0   # above: the spectrum peak clearly dominates the band


def _processor_factory(status):
    def factory():  # runs in the camera thread (MediaPipe face model + numpy/scipy)
        from ..rppg.live import HeartRateProcessor
        return HeartRateProcessor(status=status)
    return factory


class HeartPage(CameraPage):
    def __init__(self):
        super().__init__("❤️  Heart Rate",
                         "Step 5: pulse from the colour of your skin (rPPG), classic POS algorithm = the "
                         "baseline for PhysNet (Step 6). Sit still, face the light, wait 5-10 s.")
        self.track_hands = False
        self.show_lm.setText("Show skin regions")
        self.mirror.hide()          # mirroring does not matter for the pulse
        self._last = None
        self._seq = None

        self.bpm_lab = QLabel("--")
        self.bpm_lab.setObjectName("BigValue")
        self.quality = label("Start the camera", "Muted")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.wave = SignalPlot("#ff6b7a", 110)
        self.spec = SignalPlot("#4f8cff", 90, x_range=(HR_BAND[0] * 60, HR_BAND[1] * 60), x_label="BPM")

        # ---- validation against a reference device
        self.ref = QSpinBox()
        self.ref.setRange(30, 220)
        self.ref.setValue(70)
        self.ref.setSuffix(" BPM")
        self.log_btn = QPushButton("✔  Log comparison")
        self.log_btn.setObjectName("Primary")
        self.log_btn.clicked.connect(self._log)
        self.log_info = label("", "Muted")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QLabel("Smartwatch / oximeter"), 0, 0)
        grid.addWidget(self.ref, 0, 1)
        grid.addWidget(self.log_btn, 1, 0, 1, 2)
        val_w = QWidget()
        val_w.setLayout(grid)

        self.add_side(
            card("Heart rate (POS)", self.bpm_lab, self.quality, self.progress),
            card("Pulse wave (last 6 s)", self.wave),
            card("Spectrum", self.spec),
            card("Validation", label("Read your real pulse on a smartwatch or oximeter at the same time, "
                                     "then log it. Builds the MAE table for the report.", "Muted"),
                 val_w, self.log_info),
            card("Tips", label("•  Light in front of you, not behind\n•  Do not move or talk\n"
                               "•  No backlight / flickering lamp\n•  Only numbers are saved, never video",
                               "Muted")),
            width=330)
        self._update_log_info()

    def configure_worker(self, worker):
        worker.set_processor_factory(_processor_factory(self.status_message.emit))

    # ------------------------------------------------------------------ frames
    def on_frame(self, rgb, info):
        self.video.set_frame(rgb)
        if not info.get("face", False):
            self.quality.setText("⚠ No face detected")
        prog = info.get("hr_progress", 0.0)
        self.progress.setValue(int(prog * 100))
        hr = info.get("hr")
        if hr is None:
            if info.get("face"):
                self.quality.setText(f"Measuring… keep still ({prog * 100:.0f}%)")
            self.bpm_lab.setText("--")
            self._last = None
            return
        if info.get("hr_seq") == self._seq:   # the estimate is refreshed every 0.5 s, frames come faster
            return
        self._seq = info.get("hr_seq")
        self._last = hr
        good = hr["snr"] >= GOOD_SNR_DB
        self.bpm_lab.setText(f"{hr['bpm']:.0f} BPM" if hr["bpm"] is not None else "--")
        self.quality.setText(f"{'✔ Good signal' if good else '⚠ Weak signal (move less, more light)'}"
                             f"  ·  SNR {hr['snr']:+.1f} dB  ·  window {HR_WINDOW_S:.0f} s")
        self.wave.set_data(hr["wave"])
        self.spec.set_data(hr["spec"], hr["spec_bpm"], marker=hr["bpm_raw"])

    def camera_stopped(self):
        self._last = None
        self._seq = None
        self.bpm_lab.setText("--")
        self.quality.setText("Start the camera")
        self.progress.setValue(0)
        self.wave.clear()
        self.spec.clear()

    # ------------------------------------------------------------------ validation log
    def _log(self):
        if not self._last or self._last["bpm"] is None:
            self.status_message.emit("⚠ No heart-rate estimate yet: start the camera and wait")
            return
        HR_DIR.mkdir(parents=True, exist_ok=True)
        path = HR_DIR / "validation.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "method", "estimate_bpm", "reference_bpm", "abs_error", "snr_db"])
            est, ref = self._last["bpm"], self.ref.value()
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "POS", f"{est:.1f}", ref, f"{abs(est - ref):.1f}",
                        f"{self._last['snr']:.2f}"])
        self.status_message.emit(f"Logged: POS {est:.0f} vs reference {ref} BPM")
        self._update_log_info()

    def _update_log_info(self):
        path = HR_DIR / "validation.csv"
        if not path.exists():
            self.log_info.setText("No measurement logged yet.")
            return
        with open(path, newline="") as f:
            errs = [float(r["abs_error"]) for r in csv.DictReader(f) if r["method"] == "POS"]
        if errs:
            self.log_info.setText(f"{len(errs)} measurements · MAE {sum(errs) / len(errs):.1f} BPM")
