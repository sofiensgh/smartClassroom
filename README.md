# Smart Classroom

A desktop application (PySide6) built for the **NVIDIA DLI deep-learning course**.
Deep learning is the core: every module uses a network we train ourselves in PyTorch,
always compared with a simpler baseline.

| Module | What it does | Deep model | Status |
|---|---|---|---|
| ✋ **Magic Slides** | Control PowerPoint / PDF slides with hand gestures, plus an on-screen laser pointer | **LSTM** over hand-landmark sequences (vs MLP single-frame baseline, GRU, CNN+GRU on hand crops, ablation without motion features) | ✅ working |
| ❤️ **Heart Rate** | Heart rate from a normal webcam (rPPG) | PhysNet (3D CNN) vs classic POS algorithm | planned |
| 😰 **Stress Level** | Anonymous class-level stress indicator | Multimodal CNN + LSTM, late fusion | planned |
| 📊 **Dashboard** | Teacher view combining the modules | - | planned |

## Run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python main.py
```

The MediaPipe hand model downloads automatically on the first run (MediaPipe only turns
pixels into hand keypoints; the gestures are decided by our trained network).

## Gesture pipeline

1. **Record a dataset** (app → *Data & Training*): pseudonym, gesture, countdown, 30-frame
   samples of 21 hand landmarks (+ 96×96 hand crops), saved in `data/gestures/<person>/<gesture>/`.
2. **Train** (app → *Train models*, or on a GPU):
   ```bash
   python -m smartclass.training.train_gestures            # all models
   python -m smartclass.training.pack                       # zip for Colab
   ```
   Colab notebook: [`notebooks/train_gestures_colab.ipynb`](notebooks/train_gestures_colab.ipynb).
   Train/val/test split **by person**, augmentation, early stopping, fixed seed. Outputs:
   learning curves, confusion matrices, per-class metrics and a comparison table in `output/gestures/`.
3. **Use it live** (app → *Magic Slides*): sliding window → LSTM → smoothing, confidence threshold
   and cooldown → slide keys; `point` drives a laser pointer overlay.

| Gesture | Action |
|---|---|
| swipe right / left | next / previous slide |
| open palm (hold) | black screen |
| pinch | zoom in / out |
| point | laser pointer |

**Current results** (`output/gestures/results.csv`) come from **one person only** (random split),
so they prove the pipeline but are optimistic. More people will be recorded for a real
person-independent evaluation.

## Ethics

Participants opt in and are identified only by a pseudonym. The classroom modules show
class averages only, store no video, and are indicators, not a medical diagnosis.
The dataset in `data/gestures` contains only the author's own hand.

## Layout

```
smartclass/
├── camera.py, vision/        camera thread, MediaPipe hand tracking (tools)
├── data/                     dataset I/O, features shared by training and live inference
├── models/                   our PyTorch models (MLP, LSTM/GRU, CNN+GRU)
├── training/                 datasets, augmentation, training loop, metrics, figures
├── inference/                live recognition, decision logic, key sending
└── ui/                       PySide6 pages (dark theme)
```
