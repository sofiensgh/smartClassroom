"""Reusable widgets."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget


class VideoView(QLabel):
    """Shows numpy RGB frames, scaled to fit while keeping the aspect ratio."""

    def __init__(self, placeholder="Camera off"):
        super().__init__(placeholder)
        self.setObjectName("Video")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 270)
        # Ignored: the pixmap must not drive the widget size, otherwise the window grows frame after frame
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._placeholder = placeholder

    def set_frame(self, rgb):
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(img).scaled(self.contentsRect().size(), Qt.KeepAspectRatio,
                                                     Qt.SmoothTransformation))

    def clear_frame(self):
        self.clear()
        self.setText(self._placeholder)


def card(title=None, *widgets):
    box = QFrame()
    box.setObjectName("Card")
    lay = QVBoxLayout(box)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(8)
    if title:
        t = QLabel(title)
        t.setObjectName("CardTitle")
        lay.addWidget(t)
    for w in widgets:
        lay.addWidget(w)
    return box


def label(text, name=None, wrap=True):
    lab = QLabel(text)
    if name:
        lab.setObjectName(name)
    lab.setWordWrap(wrap)
    return lab


class PlaceholderPage(QWidget):
    """Page for modules that come in later steps."""

    def __init__(self, title, step, lines):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 28, 32, 28)
        lay.addWidget(label(title, "Title"))
        lay.addWidget(label(f"Coming in {step}", "Subtitle"))
        lay.addSpacing(12)
        lay.addWidget(card("What this module will do", *[label("•  " + s) for s in lines]))
        lay.addStretch()
