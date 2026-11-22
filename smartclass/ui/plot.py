"""Tiny live line plot (QPainter): much lighter than matplotlib for 2+ updates per second."""
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class SignalPlot(QWidget):
    def __init__(self, color="#ff6b7a", height=110, x_range=None, x_label=""):
        super().__init__()
        self.setMinimumHeight(height)
        self.color = QColor(color)
        self.x_range, self.x_label = x_range, x_label
        self.x = self.y = None
        self.marker = None

    def set_data(self, y, x=None, marker=None):
        """y: values; x: optional x values (else sample index); marker: an x value to highlight."""
        self.y = None if y is None else np.asarray(y, dtype=float)
        self.x = None if x is None or y is None else np.asarray(x, dtype=float)
        self.marker = marker
        self.update()

    def clear(self):
        self.set_data(None)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(4, 6, -4, -16 if self.x_label else -6)
        p.fillRect(self.rect(), QColor("#14161c"))
        p.setPen(QPen(QColor("#262a35"), 1))
        p.drawLine(r.left(), r.center().y(), r.right(), r.center().y())
        if self.y is None or len(self.y) < 2:
            p.setPen(QColor("#5d6372"))
            p.drawText(self.rect(), Qt.AlignCenter, "waiting for signal…")
            return
        x = self.x if self.x is not None else np.arange(len(self.y))
        x0, x1 = self.x_range or (x.min(), x.max())
        y0, y1 = self.y.min(), self.y.max()
        if y1 - y0 < 1e-12:
            y1 = y0 + 1
        sx = lambda v: r.left() + (v - x0) / max(x1 - x0, 1e-12) * r.width()   # noqa: E731
        sy = lambda v: r.bottom() - (v - y0) / (y1 - y0) * r.height()           # noqa: E731
        path = QPainterPath(QPointF(sx(x[0]), sy(self.y[0])))
        for xv, yv in zip(x[1:], self.y[1:]):
            path.lineTo(sx(xv), sy(yv))
        p.setPen(QPen(self.color, 2))
        p.drawPath(path)
        if self.marker is not None:
            mx = sx(self.marker)
            p.setPen(QPen(QColor("#e6e8ee"), 1, Qt.DashLine))
            p.drawLine(QPointF(mx, r.top()), QPointF(mx, r.bottom()))
        if self.x_label:
            p.setPen(QColor("#8a90a0"))
            p.drawText(r.left(), self.height() - 3, f"{x0:.0f}")
            p.drawText(r.right() - 30, self.height() - 3, f"{x1:.0f}")
            p.drawText(r.center().x() - 20, self.height() - 3, self.x_label)
        p.end()
