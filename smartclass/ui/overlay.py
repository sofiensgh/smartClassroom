"""Step 4 - transparent, click-through, always-on-top window over the presentation screen.

Draws the laser pointer (follows the index fingertip while the model says "point")
and a short toast for every action ("▶ Next slide"). It never takes focus or clicks,
so PowerPoint keeps receiving the keyboard.
"""
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QRadialGradient
from PySide6.QtWidgets import QWidget

# The part of the camera image mapped to the whole screen: the hand does not need to
# reach the image borders (where tracking is worse) to reach the screen borders.
ACTIVE_X = (0.15, 0.85)
ACTIVE_Y = (0.12, 0.72)
SMOOTH = 0.35   # EMA factor for the pointer position (lower = smoother, more lag)


class PointerOverlay(QWidget):
    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._pos = None
        self._toast = ""
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self._clear_toast)

    def set_screen(self, screen):
        self.setGeometry(screen.geometry())

    def set_pointer(self, tip):
        """tip = (x, y) of the fingertip in [0, 1] camera coordinates, or None to hide the pointer."""
        if tip is None:
            if self._pos is not None:
                self._pos = None
                self.update()
            return
        nx = (tip[0] - ACTIVE_X[0]) / (ACTIVE_X[1] - ACTIVE_X[0])
        ny = (tip[1] - ACTIVE_Y[0]) / (ACTIVE_Y[1] - ACTIVE_Y[0])
        target = QPointF(min(max(nx, 0), 1) * self.width(), min(max(ny, 0), 1) * self.height())
        if self._pos is None:
            self._pos = target
        else:
            self._pos = self._pos + (target - self._pos) * SMOOTH
        self.update()

    def toast(self, text, ms=1000):
        self._toast = text
        self._toast_timer.start(ms)
        self.update()

    def _clear_toast(self):
        self._toast = ""
        self.update()

    def paintEvent(self, _):
        if self._pos is None and not self._toast:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._pos is not None:
            g = QRadialGradient(self._pos, 26)
            g.setColorAt(0.0, QColor(255, 40, 40, 255))
            g.setColorAt(0.35, QColor(255, 40, 40, 200))
            g.setColorAt(1.0, QColor(255, 40, 40, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(g)
            p.drawEllipse(self._pos, 26, 26)
        if self._toast:
            p.setFont(QFont("Segoe UI", 20, QFont.DemiBold))
            r = p.fontMetrics().boundingRect(self._toast).adjusted(-24, -14, 24, 14)
            r.moveCenter(QRectF(0, self.height() - 120, self.width(), 60).center().toPoint())
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(20, 22, 28, 200))
            p.drawRoundedRect(r, 14, 14)
            p.setPen(QColor("white"))
            p.drawText(r, Qt.AlignCenter, self._toast)
        p.end()
