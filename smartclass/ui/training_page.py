"""Data & Training page. Step 2: record our own gesture dataset.

Flow of one sample:  idle -> countdown (3 s) -> recording (SEQ_LEN frames) -> save -> idle
"Auto" mode repeats countdown -> recording until the person has SAMPLES_PER_GESTURE samples of the gesture.
Saving is tiny (one small .npz), so it runs in the UI thread; the camera/MediaPipe work stays in CameraWorker.
"""
import csv

import cv2
import numpy as np
from PySide6.QtCore import QRegularExpression, QStringListModel, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QRegularExpressionValidator
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QCompleter, QGridLayout,
                               QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
                               QSpinBox, QTableWidget, QTableWidgetItem, QWidget)

from ..config import (CROP_SIZE, GESTURE_HELP, GESTURES, MIN_HAND_RATIO, OUTPUT_DIR, SAMPLES_PER_GESTURE,
                      SEQ_LEN)
from ..data import gesture_dataset as ds
from ..training.specs import MODEL_SPECS
from ..training.worker import TrainWorker
from .camera_page import CameraPage
from .widgets import card, label

FIRST_COUNTDOWN = 3   # seconds before the first sample
NEXT_COUNTDOWN = 2    # seconds between samples in auto mode (time to reset the hand)


class TrainingPage(CameraPage):
    def __init__(self):
        super().__init__("🧠  Data & Training",
                         "Step 2: record our own gesture dataset. Each sample = "
                         f"{SEQ_LEN} frames of hand landmarks (+ optional hand crops for the CNN).")
        self.want_crops = True
        self._state = "idle"          # idle | countdown | recording
        self._training = False
        self._auto = False
        self._count = 0
        self._buf = []
        self._rec_hand = None         # handedness locked for the current sample
        self._last_saved = None
        self._counts = {}
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

        # ---- recording card
        self.person = QLineEdit()
        self.person.setPlaceholderText("pseudonym, e.g. p01")
        self.person.setValidator(QRegularExpressionValidator(QRegularExpression(r"[A-Za-z0-9_-]{1,20}")))
        self._people_model = QStringListModel()
        self.person.setCompleter(QCompleter(self._people_model, self))
        self.person.textChanged.connect(self._refresh_table)
        self.gesture = QComboBox()
        for g, desc in GESTURES.items():
            self.gesture.addItem(f"{g}  ({desc})", g)
        self.gesture.currentIndexChanged.connect(self._gesture_changed)
        self.help = label("", "Muted")
        self.crops_cb = QCheckBox(f"Save {CROP_SIZE}×{CROP_SIZE} hand crops (for the CNN)")
        self.crops_cb.setChecked(True)
        self.crops_cb.toggled.connect(self._crops_toggled)

        self.rec_btn = QPushButton("●  Record 1")
        self.rec_btn.setObjectName("Primary")
        self.rec_btn.clicked.connect(lambda: self._begin(auto=False))
        self.auto_btn = QPushButton(f"⟳  Auto to {SAMPLES_PER_GESTURE}")
        self.auto_btn.setObjectName("Primary")
        self.auto_btn.clicked.connect(lambda: self._begin(auto=True))
        self.cancel_btn = QPushButton("■  Stop")
        self.cancel_btn.setObjectName("Danger")
        self.cancel_btn.clicked.connect(lambda: self._end_session("Recording stopped"))
        self.undo_btn = QPushButton("✕  Delete last sample")
        self.undo_btn.clicked.connect(self._delete_last)
        self.state_lab = QLabel("Ready")
        self.state_lab.setObjectName("BigValue")

        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addWidget(QLabel("Person"), 0, 0)
        form.addWidget(self.person, 0, 1)
        form.addWidget(QLabel("Gesture"), 1, 0)
        form.addWidget(self.gesture, 1, 1)
        buttons = QGridLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.rec_btn, 0, 0)
        buttons.addWidget(self.auto_btn, 0, 1)
        buttons.addWidget(self.cancel_btn, 1, 0)
        buttons.addWidget(self.undo_btn, 1, 1)
        rec_card = card("Record", _wrap(form), self.help, self.crops_cb, _wrap(buttons), self.state_lab)

        # ---- dataset card
        self.table = QTableWidget(len(GESTURES), 3)
        self.table.setHorizontalHeaderLabels(["Gesture", "This person", "All"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.setFixedHeight(self.table.verticalHeader().defaultSectionSize() * (len(GESTURES) + 1) + 6)
        for r, g in enumerate(GESTURES):
            self.table.setItem(r, 0, QTableWidgetItem(g))
        self.summary = label("", "Muted")
        data_card = card("Dataset  (data/gestures)", self.table, self.summary)

        tips = card("Protocol", label(
            "•  Use a pseudonym, never a real name. Ask for consent.\n"
            f"•  Aim for {SAMPLES_PER_GESTURE} samples per gesture and person, at least 5 people.\n"
            "•  Vary distance, position in the frame, lighting and hand (left/right).\n"
            "•  Start each gesture when the red REC border appears.", "Muted"))

        self.add_side(rec_card, data_card, self._build_train_card(), tips, width=360)
        self._gesture_changed()
        self._refresh_counts()
        self._update_buttons()

    # ------------------------------------------------------------------ Step 3: training
    def _build_train_card(self):
        self.trainer = None
        self.model_cbs = {}
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        for i, (name, spec) in enumerate(MODEL_SPECS.items()):
            cb = QCheckBox(spec["label"])
            cb.setChecked(True)
            self.model_cbs[name] = cb
            grid.addWidget(cb, i, 0, 1, 2)
        self.epochs = QSpinBox()
        self.epochs.setRange(5, 300)
        self.epochs.setValue(80)
        self.epochs.setSuffix(" epochs max")
        grid.addWidget(QLabel("Training"), len(MODEL_SPECS), 0)
        grid.addWidget(self.epochs, len(MODEL_SPECS), 1)

        self.train_btn = QPushButton("🚀  Train models")
        self.train_btn.setObjectName("Primary")
        self.train_btn.clicked.connect(self._start_training)
        self.train_stop_btn = QPushButton("■  Stop")
        self.train_stop_btn.setObjectName("Danger")
        self.train_stop_btn.setEnabled(False)
        self.train_stop_btn.clicked.connect(self.stop_training)
        btns = QGridLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.addWidget(self.train_btn, 0, 0)
        btns.addWidget(self.train_stop_btn, 0, 1)

        self.train_bar = QProgressBar()
        self.train_bar.setRange(0, 1000)
        self.train_bar.setValue(0)
        self.train_bar.setTextVisible(False)
        self.train_status = label("Split by person · early stopping · results in output/gestures", "Muted")
        self.train_status.setTextFormat(Qt.RichText)
        self.results = QTableWidget(0, 3)
        self.results.setHorizontalHeaderLabels(["Model", "Test acc", "Macro-F1"])
        self.results.verticalHeader().setVisible(False)
        self.results.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results.setSelectionMode(QAbstractItemView.NoSelection)
        self.results.setFocusPolicy(Qt.NoFocus)
        rh = self.results.horizontalHeader()
        rh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2):
            rh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.results.setFixedHeight(self.results.verticalHeader().defaultSectionSize() * (len(MODEL_SPECS) + 1) + 6)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(130)
        self.log_box.setMaximumBlockCount(500)
        open_btn = QPushButton("📂  Open figures folder")
        open_btn.clicked.connect(self._open_output)
        self._load_previous_results()
        return card("Train models  (Step 3)", _wrap(grid), _wrap(btns), self.train_bar, self.train_status,
                    self.results, self.log_box, open_btn)

    def _start_training(self):
        models = [n for n, cb in self.model_cbs.items() if cb.isChecked()]
        if not models:
            self.status_message.emit("⚠ Select at least one model")
            return
        self.stop()                                   # free the CPU and the webcam during training
        self._models_order = models
        self.trainer = TrainWorker(models, self.epochs.value())
        self.trainer.log.connect(self._train_log)
        self.trainer.progress.connect(self._train_progress)
        self.trainer.done.connect(self._train_done)
        self.trainer.failed.connect(self._train_failed)
        self._train_error = None
        self.trainer.finished.connect(self._train_finished)
        self.log_box.clear()
        self.train_bar.setValue(0)
        self.trainer.start()
        self._set_training_ui(True)
        self.status_message.emit("Training started (the first run imports PyTorch, a few seconds)...")

    def stop_training(self, wait_ms=0):
        """Ask the training thread to stop (after the current batch). Returns True if not running."""
        if not self.trainer:
            return True
        self.trainer.request_stop()
        self.train_status.setText("Stopping after the current batch…")
        return self.trainer.wait(wait_ms) if wait_ms else False

    def _set_training_ui(self, on):
        self._training = on
        self.train_btn.setEnabled(not on)
        self.train_stop_btn.setEnabled(on)
        for cb in self.model_cbs.values():
            cb.setEnabled(not on)
        self.epochs.setEnabled(not on)
        self.start_btn.setEnabled(not on)
        self._update_buttons()

    def _train_log(self, text):
        self.log_box.appendPlainText(text)

    def _train_progress(self, name, epoch, total, h):
        i = self._models_order.index(name) if name in self._models_order else 0
        self.train_bar.setValue(int(1000 * (i + epoch / total) / len(self._models_order)))
        self.train_status.setText(f"{MODEL_SPECS[name]['label']}<br>epoch {epoch}/{total} · "
                                  f"val acc {h['val_acc']:.2f} · val F1 {h['val_f1']:.2f}")

    def _train_done(self, results):
        self._show_results(results)
        self.train_bar.setValue(1000 if results else 0)
        if results:
            best = max(results, key=lambda r: r["test_macro_f1"])
            self.status_message.emit(f"✔ Training done. Best: {best['model']} (F1 {best['test_macro_f1']:.2f})")

    def _train_failed(self, msg):
        self._train_error = msg
        self.status_message.emit("⚠ Training failed: " + msg)

    def _train_finished(self):
        self.trainer.deleteLater()
        self.trainer = None
        self._set_training_ui(False)
        if self._train_error:   # keep the reason visible (the status bar message disappears after 6 s)
            self.train_status.setText(f"<span style='color:#ff7b84'>⚠ Training failed: {self._train_error}</span>")
            self.train_bar.setValue(0)
        else:
            self.train_status.setText("Figures and tables are in output/gestures")

    def _show_results(self, results):
        self.results.setRowCount(len(results))
        best = max((r["test_macro_f1"] for r in results), default=None)
        for r, res in enumerate(results):
            items = [QTableWidgetItem(res["model"]), QTableWidgetItem(f"{float(res['test_acc']):.3f}"),
                     QTableWidgetItem(f"{float(res['test_macro_f1']):.3f}")]
            if float(res["test_macro_f1"]) == float(best):
                for it in items:
                    it.setForeground(QColor("#7fe0a3"))
            for c, it in enumerate(items):
                self.results.setItem(r, c, it)

    def _load_previous_results(self):
        path = OUTPUT_DIR / "gestures" / "results.csv"
        if path.exists():
            with open(path, newline="") as f:
                self._show_results(list(csv.DictReader(f)))

    def _open_output(self):
        folder = OUTPUT_DIR / "gestures"
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ------------------------------------------------------------------ helpers
    def _person_id(self):
        return ds.clean_person_id(self.person.text())

    def _gesture_key(self):
        return self.gesture.currentData()

    def _gesture_changed(self):
        self.help.setText("How: " + GESTURE_HELP[self._gesture_key()])
        self._refresh_table()

    def _crops_toggled(self, on):
        self.want_crops = on
        self._set_worker("crops", on)

    def _update_buttons(self):
        recording = self._state != "idle"
        busy = recording or self._training
        for w in (self.rec_btn, self.auto_btn, self.person, self.gesture, self.crops_cb, self.mirror):
            w.setEnabled(not busy)
        self.undo_btn.setEnabled(not busy and self._last_saved is not None)
        self.cancel_btn.setEnabled(recording)
        if hasattr(self, "train_btn"):
            self.train_btn.setEnabled(not busy)

    # ------------------------------------------------------------------ recording state machine
    def _begin(self, auto):
        if not self._person_id():
            self.status_message.emit("⚠ Enter a person pseudonym first (e.g. p01)")
            self.person.setFocus()
            return
        if not self.running:
            self.start_camera()   # the countdown gives the camera time to open
        self._auto = auto
        self._start_countdown(FIRST_COUNTDOWN)

    def _start_countdown(self, seconds):
        self._state, self._count = "countdown", seconds
        self.state_lab.setText(f"Get ready… {seconds}")
        self._update_buttons()
        self._timer.start()

    def _tick(self):
        self._count -= 1
        if self._count > 0:
            self.state_lab.setText(f"Get ready… {self._count}")
            return
        self._timer.stop()
        self._state, self._buf, self._rec_hand = "recording", [], None
        self.state_lab.setText("● REC")

    def _end_session(self, msg=None):
        self._timer.stop()
        self._state, self._auto, self._buf = "idle", False, []
        self.state_lab.setText("Ready")
        self._update_buttons()
        if msg:
            self.status_message.emit(msg)

    def _pick_hand(self, info):
        """The hand to record: keep the same handedness during a sample, else the most confident."""
        hands = info["hands"]
        if not hands:
            return None, None
        idx = max(range(len(hands)), key=lambda i: hands[i].score)
        if self._rec_hand is None:
            self._rec_hand = hands[idx].handedness
        else:
            same = [i for i, h in enumerate(hands) if h.handedness == self._rec_hand]
            if same:
                idx = max(same, key=lambda i: hands[i].score)
        crops = info.get("crops")
        return hands[idx], (crops[idx] if crops else None)

    def _finish_sample(self, info):
        person, gesture = self._person_id(), self._gesture_key()
        n = len(self._buf)
        lms = np.zeros((n, 21, 3), np.float32)
        present = np.zeros(n, bool)
        handed = [""] * n
        crops = [None] * n if self.crops_cb.isChecked() else None
        for i, (hand, crop) in enumerate(self._buf):
            if hand is not None:
                lms[i], present[i], handed[i] = hand.landmarks, True, hand.handedness
                if crops is not None:
                    crops[i] = crop
        self._buf = []

        if gesture != "none" and present.mean() < MIN_HAND_RATIO:
            self.status_message.emit(f"⚠ Sample rejected: hand visible in only {present.mean():.0%} of the frames")
        else:
            self._last_saved = ds.save_sample(person, gesture, lms, present, handed, crops,
                                              fps=info["fps"], mirror=info["mirror"])
            self._refresh_counts()
            self.status_message.emit(f"Saved {self._last_saved.relative_to(self._last_saved.parents[3])}")

        done = self._counts.get(person, {}).get(gesture, 0) >= SAMPLES_PER_GESTURE
        if self._auto and not done:
            self._start_countdown(NEXT_COUNTDOWN)
        else:
            self._end_session(f"✔ {person} / {gesture}: {SAMPLES_PER_GESTURE} samples, done!" if done else None)

    def _delete_last(self):
        if self._last_saved and self._last_saved.exists():
            self._last_saved.unlink()
            self.status_message.emit(f"Deleted {self._last_saved.name}")
        self._last_saved = None
        self._refresh_counts()
        self._update_buttons()

    # ------------------------------------------------------------------ camera hooks
    def on_frame(self, rgb, info):
        if self._state == "recording":
            self._buf.append(self._pick_hand(info))
            if len(self._buf) >= SEQ_LEN:
                self._finish_sample(info)
        self.video.set_frame(self._overlay(rgb))

    def camera_stopped(self):
        if self._state != "idle":
            self._end_session("Recording cancelled: camera stopped")

    def _overlay(self, rgb):
        """Countdown / REC feedback drawn on the frame (rgb is a fresh array per frame, drawn in place)."""
        if self._state == "idle":
            return rgb
        h, w = rgb.shape[:2]
        if self._state == "countdown":
            txt = str(self._count)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 5, 8)
            cv2.putText(rgb, txt, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_DUPLEX, 5,
                        (255, 210, 80), 8, cv2.LINE_AA)
            cv2.putText(rgb, f"next: {self._gesture_key()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 210, 80), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(rgb, (0, 0), (w - 1, h - 1), (230, 50, 60), 10)
            cv2.putText(rgb, f"REC {self._gesture_key()}  {len(self._buf)}/{SEQ_LEN}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 50, 60), 2, cv2.LINE_AA)
            cv2.rectangle(rgb, (0, h - 12), (int(w * len(self._buf) / SEQ_LEN), h), (230, 50, 60), -1)
        return rgb

    # ------------------------------------------------------------------ counters
    def _refresh_counts(self):
        self._counts, size = ds.count_samples()
        self._people_model.setStringList(sorted(self._counts))
        total = sum(sum(v.values()) for v in self._counts.values())
        self.summary.setText(f"{len(self._counts)} people · {total} samples · {size / 1e6:.1f} MB")
        self._refresh_table()

    def _refresh_table(self):
        mine = self._counts.get(self._person_id(), {})
        current = self._gesture_key()
        for r, g in enumerate(GESTURES):
            n_me = mine.get(g, 0)
            n_all = sum(p.get(g, 0) for p in self._counts.values())
            me_item = QTableWidgetItem(f"{n_me} / {SAMPLES_PER_GESTURE}")
            if n_me >= SAMPLES_PER_GESTURE:
                me_item.setForeground(QColor("#7fe0a3"))
            self.table.setItem(r, 1, me_item)
            self.table.setItem(r, 2, QTableWidgetItem(str(n_all)))
            font = self.table.item(r, 0).font()
            font.setBold(g == current)
            self.table.item(r, 0).setFont(font)


def _wrap(layout):
    w = QWidget()
    w.setLayout(layout)
    return w
