"""Magic Slides page. Step 4: our trained gesture model controls the slides live.

Camera thread : MediaPipe landmarks -> GestureRecognizer (our LSTM, sliding window) -> probabilities
UI thread     : GestureController (smoothing, threshold, cooldown) -> keys (pyautogui) + laser overlay
"""
import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QLabel, QProgressBar,
                               QWidget)

from ..config import GESTURES
from ..inference.controller import ACTIONS, GestureController, available_models, model_scores
from ..inference.keys import send_key
from .camera_page import CameraPage
from .overlay import PointerOverlay
from .widgets import card, label


def _recognizer_factory(name):
    def factory():  # runs in the camera thread: torch is imported there, not in the UI
        from ..inference.gesture_live import GestureRecognizer
        return GestureRecognizer(name)
    return factory


class SlidesPage(CameraPage):
    def __init__(self):
        super().__init__("✋  Magic Slides",
                         "Step 4: our trained gesture model controls your slides. Start the slideshow (F5), "
                         "click on it, then use your hand.")
        self.controller = None
        self._zoom_in = True
        self.overlay = PointerOverlay()

        # ---- model
        self.model_box = QComboBox()
        self.model_box.currentIndexChanged.connect(self._model_changed)
        self.model_info = label("", "Muted")

        # ---- live prediction
        self.gesture_lab = QLabel("-")
        self.gesture_lab.setObjectName("BigValue")
        self.bars = {}
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(4)
        for r, g in enumerate(GESTURES):
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            grid.addWidget(QLabel(g), r, 0)
            grid.addWidget(bar, r, 1)
            self.bars[g] = bar
        bars_w = QWidget()
        bars_w.setLayout(grid)

        # ---- control
        self.send_keys = QCheckBox("Send keys to the slideshow")
        self.send_keys.setToolTip("→ ← B + −: sent to the PowerPoint slideshow (focused automatically), "
                                  "otherwise to the window that has the focus (PDF viewer…)")
        self.laser = QCheckBox("Laser pointer on screen")
        self.laser.setChecked(True)
        self.laser.toggled.connect(self._overlay_visibility)
        self.screen_box = QComboBox()
        for i, s in enumerate(QGuiApplication.screens()):
            self.screen_box.addItem(f"Screen {i + 1}  ({s.geometry().width()}×{s.geometry().height()})")
        self.screen_box.currentIndexChanged.connect(self._screen_changed)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.5, 0.99)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.80)
        self.threshold.valueChanged.connect(lambda v: setattr(self.controller, "threshold", v) if self.controller else None)
        self.cooldown = QDoubleSpinBox()
        self.cooldown.setRange(0.3, 3.0)
        self.cooldown.setSingleStep(0.1)
        self.cooldown.setValue(1.2)
        self.cooldown.setSuffix(" s")
        self.cooldown.valueChanged.connect(lambda v: setattr(self.controller, "cooldown", v) if self.controller else None)
        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addWidget(QLabel("Confidence"), 0, 0)
        form.addWidget(self.threshold, 0, 1)
        form.addWidget(QLabel("Cooldown"), 1, 0)
        form.addWidget(self.cooldown, 1, 1)
        form.addWidget(QLabel("Screen"), 2, 0)
        form.addWidget(self.screen_box, 2, 1)
        form_w = QWidget()
        form_w.setLayout(form)
        self.action_lab = QLabel("-")
        self.action_lab.setObjectName("CardTitle")
        self.fps_lab = label("FPS -", "Muted")

        mapping = label("<br>".join(f"<b>{g}</b> → {a['label']}" for g, a in ACTIONS.items())
                        + "<br><b>point</b> → 🔴 laser pointer", "Muted")
        self.add_side(
            card("Gesture model", self.model_box, self.model_info),
            card("Live prediction", self.gesture_lab, bars_w, self.fps_lab),
            card("Slide control", self.send_keys, self.laser, form_w, QLabel("Last action"), self.action_lab),
            card("Gestures", mapping),
            width=330)
        self.refresh_models()
        self._screen_changed(0)

    # ------------------------------------------------------------------ model
    def refresh_models(self):
        """(Re)list the trained models, e.g. after a training run."""
        current = self.model_box.currentData()
        scores = model_scores()
        items = [(n + (f"  (test acc {scores[n]:.2f})" if n in scores else ""), n) for n in available_models()]
        if items != getattr(self, "_items", None):
            self._items = items
            self.model_box.blockSignals(True)
            self.model_box.clear()
            for text, n in items:
                self.model_box.addItem(text, n)
            self.model_box.setCurrentIndex(max(self.model_box.findData(current), 0))
            self.model_box.blockSignals(False)
        has = self.model_box.count() > 0
        self.model_info.setText("Decides the gesture from the last 30 frames of hand landmarks." if has else
                                "No trained model yet: record data and train in 🧠 Data & Training.")
        if self.model_box.currentData() != current:   # only reload when the selection really changed
            self._model_changed()

    def _model_name(self):
        return self.model_box.currentData()

    def _model_changed(self, *_):
        self.controller = None
        if self.worker:
            name = self._model_name()
            self.worker.set_processor_factory(_recognizer_factory(name) if name else None)

    def configure_worker(self, worker):
        name = self._model_name()
        if name:
            worker.set_processor_factory(_recognizer_factory(name))

    # ------------------------------------------------------------------ overlay
    def _screen_changed(self, i):
        screens = QGuiApplication.screens()
        if screens:
            self.overlay.set_screen(screens[min(i, len(screens) - 1)])

    def _overlay_visibility(self, *_):
        # Late frames can still arrive while the window closes: never (re)show the overlay then,
        # or this invisible top-level window would keep the application alive.
        show = (self.running and self.laser.isChecked() and self.isVisible()
                and not self.worker._stop_requested)
        self.overlay.setVisible(show)

    def stop(self, timeout_ms=5000):
        self.overlay.set_pointer(None)
        self.overlay.hide()
        return super().stop(timeout_ms)

    # ------------------------------------------------------------------ frames
    def on_frame(self, rgb, info):
        self.video.set_frame(rgb)
        self.fps_lab.setText(f"FPS {info['fps']:.0f}  ·  hands {len(info['hands'])}")
        if not self.overlay.isVisible() and self.laser.isChecked():
            self._overlay_visibility()
        probs = info.get("probs")
        if probs is None:
            return
        classes = info["classes"]
        if self.controller is None or self.controller.classes != classes:
            self.controller = GestureController(classes, self.threshold.value(), self.cooldown.value())
        out = self.controller.update(probs, time.monotonic())
        for g, p in zip(classes, probs):
            if g in self.bars:
                self.bars[g].setValue(int(p * 100))
        shown = out["active"] if out["active"] != "none" else "none"
        self.gesture_lab.setText(f"{shown}  {out['conf']:.0%}" if shown != "none" else "none")
        self.overlay.set_pointer(info.get("tip") if out["pointing"] and self.laser.isChecked() else None)
        if out["event"]:
            self._do_action(out["event"])

    def _do_action(self, gesture):
        action = ACTIONS[gesture]
        keys = action["keys"]
        key = keys[0]
        if len(keys) > 1:                      # pinch: zoom in, then zoom out, ...
            key = keys[0] if self._zoom_in else keys[1]
            self._zoom_in = not self._zoom_in
        text = action["label"] + (f"  ({'in' if key == '+' else 'out'})" if len(keys) > 1 else "")
        self.action_lab.setText(text)
        if self.laser.isChecked():
            self.overlay.toast(text)
        if self.send_keys.isChecked():
            try:
                target = send_key(key)
                self.status_message.emit(f"{text}  →  key '{key}' sent to {target}")
            except Exception as e:  # noqa: BLE001 - shown to the user
                self.status_message.emit(f"⚠ Could not send the key: {e}")
        # start a fresh window so the same gesture is not detected twice
        if self.worker and self.worker.processor is not None:
            self.worker.processor.request_reset()
        self.controller.reset()

    def camera_stopped(self):
        self.controller = None
        self.overlay.set_pointer(None)
        self.overlay.hide()
        self.gesture_lab.setText("-")
        self.fps_lab.setText("FPS -")
        for bar in self.bars.values():
            bar.setValue(0)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_models()   # pick up models trained since the last visit
