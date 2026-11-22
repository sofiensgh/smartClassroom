"""Main window: sidebar navigation + pages."""
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QHBoxLayout, QListWidget, QMainWindow, QStackedWidget, QWidget

from ..config import APP_NAME, APP_VERSION
from .home_page import HomePage
from .slides_page import SlidesPage
from .training_page import TrainingPage
from .widgets import PlaceholderPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  ·  {APP_VERSION}")
        self.resize(1280, 780)
        self.setMinimumSize(QSize(1000, 640))

        self.slides = SlidesPage()
        self.training = TrainingPage()
        pages = [
            ("🏠   Home", HomePage()),
            ("✋   Magic Slides", self.slides),
            ("❤️   Heart Rate", PlaceholderPage("❤️  Heart Rate", "Steps 5-6", [
                "Detect the face and track forehead + cheek regions",
                "Classic baseline: POS algorithm + band-pass filter + FFT",
                "Deep model: PhysNet (3D CNN) trained on UBFC-rPPG",
                "Live BPM, pulse waveform, comparison with a smartwatch"])),
            ("😰   Stress", PlaceholderPage("😰  Stress Level", "Step 7", [
                "Heart-rate variability from the pulse signal",
                "Facial cues: brows, mouth tension, blinking",
                "Multimodal deep model (CNN + LSTM + fusion) trained on UBFC-Phys",
                "Score 0-100: relaxed / normal / stressed"])),
            ("📊   Dashboard", PlaceholderPage("📊  Teacher Dashboard", "Step 8", [
                "Gesture control status and current slide",
                "Anonymous class-average stress over time",
                "Alerts: 'the class looks stressed, slow down'"])),
            ("🧠   Data & Training", self.training),
        ]
        self.camera_pages = [self.slides, self.training]

        self.nav = QListWidget()
        self.nav.setObjectName("Sidebar")
        self.nav.setFixedWidth(220)
        self.stack = QStackedWidget()
        for name, page in pages:
            self.nav.addItem(name)
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self._switch_page)
        self.nav.setCurrentRow(0)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.nav)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        for page in self.camera_pages:
            page.status_message.connect(lambda m: self.statusBar().showMessage(m, 6000))
        self.statusBar().showMessage("Ready")

    def _switch_page(self, row):
        # Only one page can hold the webcam: stop the camera of the page we leave.
        new = self.stack.widget(row)
        for page in self.camera_pages:
            if page is not new:
                page.stop()
        self.stack.setCurrentIndex(row)

    def closeEvent(self, event):
        # Release the camera cleanly. If the thread is still busy (e.g. first-run model
        # download), wait longer: destroying a running QThread crashes the app.
        trainer = self.training.trainer
        if trainer is not None and not self.training.stop_training(wait_ms=15000):  # stops after the batch
            trainer.terminate()                                # last resort, never leave a running thread
            trainer.wait()
        for page in self.camera_pages:
            if not page.stop(5000):
                self.statusBar().showMessage("Waiting for the camera thread to stop...")
                page.stop(30000)
        self.slides.overlay.close()   # top-level click-through window: close it explicitly
        super().closeEvent(event)
