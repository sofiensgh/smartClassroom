"""Heart Rate page. Step 5: classic rPPG baseline (POS). Step 6b: our PhysNet (3D CNN) live, side by side.

Camera thread : FaceLandmarker -> forehead/cheek ROIs -> mean RGB -> POS -> band-pass -> FFT peak (BPM)
                               -> 72x72 face crop -> PhysNet (own thread, every 1.5 s) -> FFT peak (BPM)
UI thread     : both BPMs, pulse waves, spectrum; logs both against a reference device (smartwatch).
"""
import csv
import time
from collections import deque
from statistics import median

from PySide6.QtWidgets import QComboBox, QGridLayout, QLabel, QProgressBar, QPushButton, QSpinBox, QWidget

from ..config import HR_BAND, HR_DIR, HR_WINDOW_S, MODELS_DIR
from .camera_page import CameraPage
from .plot import SignalPlot
from .widgets import card, label

GOOD_SNR_DB = 0.0   # above: the spectrum peak clearly dominates the band


def _processor_factory(status, phys_path=None):
    def factory():  # runs in the camera thread (MediaPipe face model + numpy/scipy + torch)
        from ..rppg.live import HeartRateProcessor
        return HeartRateProcessor(status=status, phys_path=phys_path)
    return factory


def physnet_files():
    """models/physnet*.pt: physnet.pt = latest training (v2), physnet_v1.pt = before webcam augmentation."""
    return sorted(MODELS_DIR.glob("physnet*.pt"), key=lambda p: (p.stem != "physnet", p.stem))


class HeartPage(CameraPage):
    def __init__(self):
        super().__init__("❤️  Heart Rate",
                         "Pulse from the colour of your skin (rPPG): classic POS algorithm (baseline) vs our "
                         "PhysNet 3D CNN trained on UBFC-rPPG. Sit still, face the light, wait 5-10 s.")
        self.track_hands = False
        self.lock_exposure = True   # auto exposure steps = fake pulse (measured), see CameraWorker
        self.show_lm.setText("Show skin regions")
        self.mirror.hide()          # mirroring does not matter for the pulse
        self._last = None
        self._seq = None
        self._phys = None
        self._phys_seq = None
        self._snrs = deque(maxlen=6)   # label = median of the last 3 s, so it does not flicker

        self.bpm_lab = QLabel("--")
        self.bpm_lab.setObjectName("BigValue")
        self.quality = label("Start the camera", "Muted")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.wave = SignalPlot("#ff6b7a", 110)
        self.spec = SignalPlot("#4f8cff", 90, x_range=(HR_BAND[0] * 60, HR_BAND[1] * 60), x_label="BPM")

        # ---- PhysNet (deep model, Step 6)
        self.phys_lab = QLabel("--")
        self.phys_lab.setObjectName("BigValue")
        self.phys_info = label("Start the camera", "Muted")
        self.phys_wave = SignalPlot("#3ddc97", 90)
        self.phys_model = QComboBox()
        for p in physnet_files():
            self.phys_model.addItem(p.name + ("  (default)" if p.stem == "physnet" else ""), str(p))
        self.phys_model.setToolTip("Compare PhysNet versions live (v1 = trained without webcam augmentation)")
        self.phys_model.currentIndexChanged.connect(self._model_changed)

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
            card("Heart rate · POS (classic)", self.bpm_lab, self.quality, self.progress),
            card("Heart rate · PhysNet (deep, ours)", self.phys_model, self.phys_lab, self.phys_info, self.phys_wave),
            card("POS pulse wave (last 6 s)", self.wave),
            card("POS spectrum", self.spec),
            card("Validation", label("Read your real pulse on a smartwatch or oximeter at the same time, "
                                     "then log it. Logs POS and PhysNet: MAE table for the report.", "Muted"),
                 val_w, self.log_info),
            card("Tips", label("•  Light in front of you, not behind\n•  Do not move or talk\n"
                               "•  No backlight / flickering lamp\n•  Only numbers are saved, never video",
                               "Muted")),
            width=330)
        self._update_log_info()

    def configure_worker(self, worker):
        worker.set_processor_factory(_processor_factory(self.status_message.emit, self.phys_model.currentData()))

    def _model_changed(self):
        self._reset_physnet("Loading model…")
        if self.worker is not None:   # rebuilt in the camera thread (POS restarts too)
            self.configure_worker(self.worker)

    # ------------------------------------------------------------------ frames
    def on_frame(self, rgb, info):
        self.video.set_frame(rgb)
        self._on_physnet(info)
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
        self._snrs.append(hr["snr"])
        good = median(self._snrs) >= GOOD_SNR_DB
        self.bpm_lab.setText(f"{hr['bpm']:.0f} BPM" if hr["bpm"] is not None else "--")
        self.quality.setText(f"{'✔ Good signal' if good else '⚠ Weak signal (move less, more light)'}"
                             f"  ·  SNR {median(self._snrs):+.1f} dB  ·  window {HR_WINDOW_S:.0f} s")
        self.wave.set_data(hr["wave"])
        self.spec.set_data(hr["spec"], hr["spec_bpm"], marker=hr["bpm_raw"])

    def _on_physnet(self, info):
        if info.get("phys_error"):
            self.phys_info.setText("⚠ " + info["phys_error"])
            return
        res = info.get("phys")
        if res is None:
            self._phys = None
            self.phys_lab.setText("--")
            if info.get("face"):
                self.phys_info.setText(f"Collecting face frames… ({info.get('phys_progress', 0) * 100:.0f}%)")
            return
        if info.get("phys_seq") == self._phys_seq:
            return
        self._phys_seq = info.get("phys_seq")
        if res.get("error"):
            self.phys_info.setText("⚠ " + res["error"])
            return
        self._phys = res
        self.phys_lab.setText(f"{res['bpm']:.0f} BPM")
        diff = f"  ·  Δ POS {res['bpm'] - self._last['bpm']:+.0f}" if self._last and self._last["bpm"] else ""
        self.phys_info.setText(f"{res.get('model', 'PhysNet')}  ·  SNR {res['snr']:+.1f} dB{diff}")
        self.phys_wave.set_data(res["wave"])

    def camera_stopped(self):
        self._last = None
        self._seq = None
        self._snrs.clear()
        self._reset_physnet("Start the camera")

    def _reset_physnet(self, text):
        self._phys = None
        self._phys_seq = None
        self.phys_lab.setText("--")
        self.phys_info.setText(text)
        self.phys_wave.clear()
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
            now, ref = time.strftime("%Y-%m-%d %H:%M:%S"), self.ref.value()
            msg = []
            for method, res in (("POS", self._last), ((self._phys or {}).get("model", "PhysNet"), self._phys)):
                if res and res.get("bpm") is not None:
                    est = res["bpm"]
                    w.writerow([now, method, f"{est:.1f}", ref, f"{abs(est - ref):.1f}", f"{res['snr']:.2f}"])
                    msg.append(f"{method} {est:.0f}")
        self.status_message.emit(f"Logged: {', '.join(msg)} vs reference {ref} BPM")
        self._update_log_info()

    def _update_log_info(self):
        path = HR_DIR / "validation.csv"
        if not path.exists():
            self.log_info.setText("No measurement logged yet.")
            return
        errs = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                errs.setdefault(r["method"], []).append(float(r["abs_error"]))
        if errs:
            self.log_info.setText("\n".join(f"{m}: {len(e)} measurements · MAE {sum(e) / len(e):.1f} BPM"
                                            for m, e in errs.items()))
