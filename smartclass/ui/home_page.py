"""Home page: project overview, the 3 deep-learning models, roadmap and ethics."""
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from .widgets import card, label

MODULES = [
    ("✋  Magic Slides", "Control your slides with hand gestures.",
     "Gesture model: MLP baseline vs LSTM on landmark sequences vs CNN on hand images"),
    ("❤️  Heart Rate", "Measure the heart rate from a normal webcam (rPPG).",
     "PhysNet (3D CNN) trained on UBFC-rPPG, compared with the classic POS method"),
    ("😰  Stress Level", "Estimate stress from heart signal + facial cues.",
     "Multimodal network: CNN (face) + LSTM (pulse signal) + fusion, trained on UBFC-Phys"),
]

ROADMAP = [
    ("Step 0", "Desktop app skeleton + environment", True),
    ("Step 1", "Hand tracking (live camera)", True),
    ("Step 2", "Gesture dataset recorder", True),
    ("Step 3", "Train gesture models (MLP / LSTM / CNN): first run on 1 person, add more people", True),
    ("Step 4", "Magic Slides controls PowerPoint (live LSTM + laser pointer)", True),
    ("Step 5", "Heart rate: classic POS baseline (live BPM, pulse wave, validation log)", True),
    ("Step 6", "Heart rate: PhysNet deep model (UBFC-rPPG, live next to POS)", True),
    ("Step 7", "Stress: multimodal deep model", False),
    ("Step 8", "Teacher dashboard", False),
    ("Step 9", "Evaluation + report figures", False),
]


class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(32, 28, 32, 28)
        lay.setSpacing(16)
        lay.addWidget(label("Smart Classroom", "Title"))
        lay.addWidget(label("Teach with your hands, and feel the pulse of your class. "
                            "A deep-learning project with 3 models trained from scratch.", "Subtitle"))

        grid = QGridLayout()
        grid.setSpacing(14)
        for i, (title, desc, dl) in enumerate(MODULES):
            dl_lab = label("🧠  " + dl, "Muted")
            grid.addWidget(card(title, label(desc), dl_lab), 0, i)
        lay.addLayout(grid)

        road = []
        for step, text, done in ROADMAP:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            badge = QLabel("done" if done else "todo")
            badge.setObjectName("BadgeDone" if done else "Badge")
            h.addWidget(badge)
            h.addWidget(QLabel(f"<b>{step}</b> &nbsp; {text}"))
            h.addStretch()
            road.append(row)

        ethics = [label(t) for t in (
            "•  Students opt in before any measurement.",
            "•  The teacher only sees the class average, never a named student.",
            "•  No video is stored: frames are processed live, only numbers are kept.",
            "•  Stress and heart rate are indicators, not a medical diagnosis.",
        )]
        row = QHBoxLayout()
        row.setSpacing(14)
        row.addWidget(card("Roadmap", *road), 3)
        row.addWidget(card("Ethics by design", *ethics), 2)
        lay.addLayout(row)
        lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
