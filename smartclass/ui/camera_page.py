"""Base class for pages with a live camera (Magic Slides, Data & Training).

Owns the camera controls and the CameraWorker thread. Subclasses override:
    on_frame(rgb, info)   called in the UI thread for every frame
    camera_stopped()      called when the camera thread has finished
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QPushButton, QScrollArea, QVBoxLayout,
                               QWidget)

from ..camera import CameraWorker
from .widgets import VideoView, label


class CameraPage(QWidget):
    status_message = Signal(str)

    def __init__(self, title, subtitle):
        super().__init__()
        self.worker = None
        self.want_crops = False

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        root.addWidget(label(title, "Title"))
        root.addWidget(label(subtitle, "Subtitle"))

        # ---- camera controls
        bar = QHBoxLayout()
        self.cam_box = QComboBox()
        self.cam_box.addItems([f"Camera {i}" for i in range(4)])
        self.mirror = QCheckBox("Mirror")
        self.mirror.setChecked(True)
        self.show_lm = QCheckBox("Show landmarks")
        self.show_lm.setChecked(True)
        self.show_lm.toggled.connect(lambda on: self._set_worker("draw", on))
        self.mirror.toggled.connect(lambda on: self._set_worker("mirror", on))
        self.start_btn = QPushButton("▶  Start camera")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self.toggle)
        for w in (self.cam_box, self.mirror, self.show_lm):
            bar.addWidget(w)
        bar.addStretch()
        bar.addWidget(self.start_btn)
        root.addLayout(bar)

        # ---- video + side panel (added by the subclass with add_side)
        self.body = QHBoxLayout()
        self.body.setSpacing(14)
        self.video = VideoView("Press  ▶ Start camera")
        self.body.addWidget(self.video, 3)
        root.addLayout(self.body, 1)

    def add_side(self, *widgets, width=300):
        inner = QWidget()
        side = QVBoxLayout(inner)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(12)
        for w in widgets:
            side.addWidget(w)
        side.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        scroll.setFixedWidth(width + 14)
        self.body.addWidget(scroll)

    # ------------------------------------------------------------------ camera
    @property
    def running(self):
        return self.worker is not None

    def _set_worker(self, attr, value):
        if self.worker:
            setattr(self.worker, attr, value)

    def toggle(self):
        if self.worker:
            self.stop()
        else:
            self.start_camera()

    def start_camera(self):
        if self.worker:
            return
        w = CameraWorker(self.cam_box.currentIndex(), self.mirror.isChecked(), track_hands=True,
                         draw=self.show_lm.isChecked(), crops=self.want_crops)
        w.frame_ready.connect(self.on_frame)
        w.status.connect(self.status_message.emit)
        w.error.connect(lambda msg: self.status_message.emit("⚠ " + msg))
        w.finished.connect(lambda w=w: self._on_finished(w))
        self.configure_worker(w)
        self.worker = w
        w.start()
        self.cam_box.setEnabled(False)
        self._set_button("■  Stop camera", "Danger")
        self.status_message.emit("Opening camera...")

    def stop(self, timeout_ms=5000):
        """Stop the camera thread. Returns False if it is still running after the timeout."""
        if not self.worker:
            return True
        self.start_btn.setEnabled(False)  # no double clicks while the thread shuts down
        return self.worker.stop(timeout_ms)

    def _on_finished(self, w):
        w.deleteLater()
        if w is not self.worker:
            return
        self.worker = None
        self.video.clear_frame()
        self.cam_box.setEnabled(True)
        self.start_btn.setEnabled(True)
        self._set_button("▶  Start camera", "Primary")
        self.camera_stopped()

    def _set_button(self, text, style_name):
        self.start_btn.setText(text)
        self.start_btn.setObjectName(style_name)
        st = self.start_btn.style()  # re-apply the stylesheet after the object name changed
        st.unpolish(self.start_btn)
        st.polish(self.start_btn)

    # ------------------------------------------------------------------ hooks
    def configure_worker(self, worker):
        """Called before the camera thread starts (e.g. to attach a processor factory)."""

    def on_frame(self, rgb, info):
        self.video.set_frame(rgb)

    def camera_stopped(self):
        pass
