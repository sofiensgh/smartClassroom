"""Runs gesture training in a QThread so the UI stays responsive."""
import traceback

from PySide6.QtCore import QThread, Signal


class TrainWorker(QThread):
    log = Signal(str)
    progress = Signal(str, int, int, dict)   # model name, epoch, total epochs, metrics of that epoch
    done = Signal(list)                      # list of result dicts (one per trained model)
    failed = Signal(str)

    def __init__(self, models, epochs, parent=None):
        super().__init__(parent)
        self.models, self.epochs = models, epochs
        self._stop = False

    def request_stop(self):
        self._stop = True   # checked after every batch

    def run(self):
        try:
            from .train_gestures import TrainConfig, run  # imports torch: only in this thread
            results = run(TrainConfig(models=self.models, epochs=self.epochs), log=self.log.emit,
                          progress=lambda n, e, t, h: self.progress.emit(n, e, t, h),
                          should_stop=lambda: self._stop)
            self.done.emit(results)
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(e))


class StressTrainWorker(QThread):
    """Step 7: trains the face / pulse / fusion stress models (cross-validation by session) in a thread."""
    log = Signal(str)
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, epochs=30, parent=None):
        super().__init__(parent)
        self.epochs = epochs

    def run(self):
        try:
            from .train_stress import run  # imports torch: only in this thread
            self.done.emit(run(epochs=self.epochs, log=self.log.emit))
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(e))
