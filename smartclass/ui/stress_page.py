"""Stress page (Step 7): pilot dataset recorder + multimodal model training + live stress indicator.

Protocol (like UBFC-Phys T1 vs T3), 2 min per session, alternate rest / stress:
  REST   : sit, breathe slowly with the guide
  STRESS : timed mental arithmetic (3 choices, 4 s each, beep when wrong or too slow)
Only numbers are stored (52 blendshapes + skin colour per frame), never video.
"""
import random
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QStackedWidget, QWidget)

from ..config import STRESS_SESSION_S
from .camera_page import CameraPage
from .widgets import card, label

QUESTION_S = 4.0


def _processor_factory(control, status):
    def factory():   # camera thread: MediaPipe + torch
        from ..stress.live import StressProcessor
        return StressProcessor(control, status=status)
    return factory


class StressPage(CameraPage):
    def __init__(self):
        super().__init__("😰  Stress Level",
                         "Step 7 (pilot): multimodal deep model, LSTM on facial expressions + 1D CNN on the pulse "
                         "wave, fused. Indicator only, never a diagnosis; only numbers are stored, never video.")
        from ..stress.live import StressControl
        self.track_hands = False
        self.lock_exposure = True      # the pulse branch needs a stable image (see Heart Rate)
        self.mirror.hide()
        self.show_lm.setText("Show skin regions")
        self.control = StressControl()
        self.recording = None          # label while a session is running
        self.trainer = None
        self._seq = None

        # ---- live indicator
        self.score = QLabel("--")
        self.score.setObjectName("BigValue")
        self.level = label("Start the camera", "Muted")
        self.fill = QProgressBar()
        self.fill.setRange(0, 100)
        self.fill.setTextVisible(False)
        self.branches = label("", "Muted")

        # ---- recorder
        self.person = QLineEdit("p01")
        self.cond = QComboBox()
        self.cond.addItem("rest  (relax, breathe slowly)", "rest")
        self.cond.addItem("stress  (timed mental arithmetic)", "stress")
        self.dur = QSpinBox()
        self.dur.setRange(30, 300)
        self.dur.setValue(STRESS_SESSION_S)
        self.dur.setSuffix(" s")
        self.rec_btn = QPushButton("●  Start session")
        self.rec_btn.setObjectName("Primary")
        self.rec_btn.clicked.connect(self._toggle_session)
        self.rec_bar = QProgressBar()
        self.rec_bar.setRange(0, 100)
        self.rec_bar.setTextVisible(False)
        self.counts = label("", "Muted")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QLabel("Person"), 0, 0)
        grid.addWidget(self.person, 0, 1)
        grid.addWidget(QLabel("Condition"), 1, 0)
        grid.addWidget(self.cond, 1, 1)
        grid.addWidget(QLabel("Duration"), 2, 0)
        grid.addWidget(self.dur, 2, 1)
        rec_w = QWidget()
        rec_w.setLayout(grid)

        # ---- task panel (idle / breathing / arithmetic)
        self.task = QStackedWidget()
        self.task.addWidget(label("The task appears here during a session.", "Muted"))
        self.breath = QLabel("Breathe in…")
        self.breath.setObjectName("BigValue")
        self.breath.setAlignment(Qt.AlignCenter)
        self.task.addWidget(self.breath)
        arith = QWidget()
        al = QGridLayout(arith)
        al.setContentsMargins(0, 0, 0, 0)
        self.question = QLabel("")
        self.question.setObjectName("BigValue")
        self.question.setAlignment(Qt.AlignCenter)
        self.q_bar = QProgressBar()
        self.q_bar.setRange(0, 1000)
        self.q_bar.setTextVisible(False)
        self.answers = [QPushButton("") for _ in range(3)]
        row = QHBoxLayout()
        for b in self.answers:
            b.clicked.connect(lambda _=False, b=b: self._answer(int(b.text())))
            row.addWidget(b)
        self.q_info = label("", "Muted")
        al.addWidget(self.question, 0, 0)
        al.addWidget(self.q_bar, 1, 0)
        al.addLayout(row, 2, 0)
        al.addWidget(self.q_info, 3, 0)
        self.task.addWidget(arith)

        # ---- training
        self.train_btn = QPushButton("🧠  Train stress models")
        self.train_btn.clicked.connect(self._train)
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setFixedHeight(140)

        self.add_side(
            card("Stress level (fusion model)", self.score, self.level, self.fill, self.branches),
            card("Record a session", label("Alternate rest / stress, at least 3 of each (2 min). Keep the "
                                           "same place and light.", "Muted"), rec_w, self.rec_btn, self.rec_bar,
                 self.counts),
            card("Session task", self.task),
            card("Train", label("Cross-validation by session: face only vs pulse only vs fusion.", "Muted"),
                 self.train_btn, self.train_log),
            card("Model", label("face: 52 blendshapes × 200 steps (10 Hz) → LSTM 64\n"
                                "pulse: POS wave 600 samples (30 Hz) → 1D CNN (16-32-64)\n"
                                "fusion: concat 128 → MLP 32 → P(stress)\n"
                                "score = 100 × P(stress), smoothed", "Muted")),
            width=340)

        self._tick = QTimer(self)
        self._tick.setInterval(50)
        self._tick.timeout.connect(self._on_tick)
        self._t = 0.0
        self._q_left = 0.0
        self._right = self._asked = 0
        self._update_counts()

    # ------------------------------------------------------------------ camera
    def configure_worker(self, worker):
        worker.set_processor_factory(_processor_factory(self.control, self.status_message.emit))

    def on_frame(self, rgb, info):
        self.video.set_frame(rgb)
        if info.get("stress_error"):
            self.branches.setText("⚠ " + info["stress_error"])
        if "rec_elapsed" in info:
            self.rec_bar.setValue(int(100 * info["rec_elapsed"] / info["rec_dur"]))
        if info.get("rec_saved"):
            p = Path(info["rec_saved"])
            self._end_session(f"✔ Saved {p.parent.name}/{p.name} (face {info['rec_face']:.0%})")
        elif info.get("rec_failed"):
            self._end_session("⚠ " + info["rec_failed"])
        self._show_stress(info)

    def _show_stress(self, info):
        if not info.get("models"):
            self.score.setText("--")
            self.level.setText("No stress model yet: record sessions, then Train." if info.get("face", True)
                               else "⚠ No face detected")
            return
        self.fill.setValue(int(100 * info.get("stress_fill", 0)))
        res = info.get("stress")
        if res is None:
            self.score.setText("--")
            self.level.setText("⚠ No face detected" if not info.get("face") else "Collecting 20 s of signal…")
            return
        if info.get("stress_seq") == self._seq:
            return
        self._seq = info["stress_seq"]
        s = res["score"]
        self.score.setText(f"{s:.0f} / 100")
        self.level.setText("😌 relaxed" if s < 35 else "😐 normal" if s < 65 else "😰 stressed")
        self.branches.setText("  ·  ".join(f"{m} {100 * p:.0f}" for m, p in sorted(res["probs"].items())))

    def camera_stopped(self):
        if self.recording:
            self._end_session("⚠ Camera stopped: session not saved")
        self.score.setText("--")
        self.level.setText("Start the camera")
        self.branches.setText("")
        self.fill.setValue(0)
        self._seq = None

    # ------------------------------------------------------------------ sessions
    def _toggle_session(self):
        if self.recording:
            self.control.stop()
            self._end_session("Session cancelled (not saved)")
            return
        if not self.running:
            self.status_message.emit("Start the camera first")
            return
        person = self.person.text().strip() or "p01"
        self.recording = self.cond.currentData()
        self.control.task_score = None
        self.control.start(person, self.recording, self.dur.value())
        self.rec_btn.setText("■  Cancel session")
        self.rec_btn.setObjectName("Danger")
        self._restyle(self.rec_btn)
        for w in (self.person, self.cond, self.dur, self.train_btn):
            w.setEnabled(False)
        self._t, self._right, self._asked = 0.0, 0, 0
        if self.recording == "stress":
            self.task.setCurrentIndex(2)
            self._new_question()
        else:
            self.task.setCurrentIndex(1)
        self._tick.start()

    def _end_session(self, msg):
        self._tick.stop()
        self.recording = None
        self.task.setCurrentIndex(0)
        self.rec_btn.setText("●  Start session")
        self.rec_btn.setObjectName("Primary")
        self._restyle(self.rec_btn)
        for w in (self.person, self.cond, self.dur, self.train_btn):
            w.setEnabled(True)
        self.rec_bar.setValue(0)
        self.status_message.emit(msg)
        self._update_counts(msg)

    def _on_tick(self):
        self._t += 0.05
        if self.recording == "rest":            # 4 s in, 6 s out = 6 breaths per minute
            ph = self._t % 10
            self.breath.setText("Breathe in…" if ph < 4 else "Breathe out…")
            return
        self._q_left -= 0.05
        self.q_bar.setValue(max(0, int(1000 * self._q_left / QUESTION_S)))
        if self._q_left <= 0:
            self._feedback(False, "⏱ Too slow!")

    def _new_question(self):
        a, b = random.randint(300, 999), random.choice([13, 17, 19, 23, 27, 29])
        self._answer_ok = a - b
        opts = {self._answer_ok, self._answer_ok + random.choice([-10, 10]), self._answer_ok + random.choice([-2, -1, 1, 2])}
        while len(opts) < 3:
            opts.add(self._answer_ok + random.randint(-20, 20))
        for btn, v in zip(self.answers, random.sample(sorted(opts), 3)):
            btn.setText(str(v))
        self.question.setText(f"{a} − {b} = ?")
        self._q_left = QUESTION_S

    def _answer(self, v):
        if self.recording != "stress":
            return
        ok = v == self._answer_ok
        self._feedback(ok, "✔ Right" if ok else "✘ Wrong!")

    def _feedback(self, ok, msg):
        self._asked += 1
        self._right += ok
        if not ok:
            QApplication.beep()
        self.control.task_score = self._right / self._asked
        self.q_info.setText(f"{msg}   ·   score {self._right}/{self._asked}")
        self._new_question()

    def _update_counts(self, extra=""):
        from ..stress.dataset import count_sessions
        c = count_sessions()
        txt = "\n".join(f"{p}: rest {v['rest']} · stress {v['stress']}" for p, v in sorted(c.items()))
        self.counts.setText((txt or "No session recorded yet.") + (f"\n{extra}" if extra else ""))

    # ------------------------------------------------------------------ training
    def _train(self):
        if self.trainer is not None:
            return
        from ..training.worker import StressTrainWorker
        self.train_log.clear()
        self.trainer = StressTrainWorker()
        self.trainer.log.connect(self.train_log.appendPlainText)
        self.trainer.done.connect(self._trained)
        self.trainer.failed.connect(lambda e: self.train_log.appendPlainText(f"FAILED: {e}"))
        self.trainer.finished.connect(self._train_finished)
        self.train_btn.setEnabled(False)
        self.rec_btn.setEnabled(False)
        self.trainer.start()

    def _trained(self, results):
        self.train_log.appendPlainText("Done. Figures in output/stress/. Live models reloaded.")
        if self.worker is not None:      # rebuild the processor in the camera thread with the new models
            self.configure_worker(self.worker)

    def _train_finished(self):
        self.trainer.deleteLater()
        self.trainer = None
        self.train_btn.setEnabled(True)
        self.rec_btn.setEnabled(True)

    @staticmethod
    def _restyle(w):
        st = w.style()
        st.unpolish(w)
        st.polish(w)
