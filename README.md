# Smart Classroom

A desktop application (PySide6) built for the **NVIDIA DLI deep-learning course**.
Deep learning is the core: every module uses a network we train ourselves in PyTorch,
always compared with a simpler baseline.

| Module | What it does | Deep model (ours) | Baseline | Status |
|---|---|---|---|---|
| ✋ **Magic Slides** | Control PowerPoint / PDF slides with hand gestures, plus an on-screen laser pointer | **LSTM** over hand-landmark sequences (+ GRU, CNN+GRU on hand crops, ablation without motion) | MLP on a single frame | ✅ working |
| ❤️ **Heart Rate** | Heart rate from a normal webcam (rPPG) | **PhysNet** (3D CNN) trained on UBFC-rPPG | POS algorithm (Wang 2017) | ✅ working |
| 😰 **Stress Level** | Stress indicator 0–100 (relaxed / normal / stressed) | **Multimodal**: LSTM on facial expressions + 1D CNN on the pulse wave → fusion | face only / pulse only | ✅ pilot |
| 📊 **Dashboard** | Teacher view combining the modules | - | - | planned |

## Run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python main.py
```

The MediaPipe hand and face models download automatically on the first run. MediaPipe and the
signal-processing steps are **tools only**: they turn pixels into keypoints or colour traces, and the
decisions (gesture, pulse wave, stress) come from our trained networks.

## 1. Gestures (Magic Slides)

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

**Results** (test set, one person, random split): MLP 1.00, LSTM 0.95, GRU 0.84, CNN+GRU 0.84.
Live, the LSTM is the more stable model: replaying the recorded samples gives 97/99 correct and
**0 double triggers**, against **6** for the single-frame MLP. With one person these numbers prove
the pipeline but are optimistic; more people are needed for a person-independent evaluation.

## 2. Heart rate (rPPG): PhysNet vs POS

- **POS** (classic, no learning): forehead + cheek colour → POS projection → band-pass → FFT peak.
- **PhysNet** (3D CNN, 768 k parameters, our implementation): 128 frames of a 72×72 face crop →
  pulse wave. Negative-Pearson loss, trained on **UBFC-rPPG** (42 subjects, split **by subject**
  28 / 6 / 8) on a Colab T4: [`notebooks/train_physnet_colab.ipynb`](notebooks/train_physnet_colab.ipynb).

```bash
python -m smartclass.rppg.preprocess --ubfc /path/to/DATASET_2      # face crops + PPG cache
python -m smartclass.training.train_physnet --cache data/ubfc_cache   # train + evaluate
python -m smartclass.training.train_physnet --eval-only models/physnet_v1.pt
```

**Test results** (8 unseen subjects, 10 s windows, MAE in BPM), from `output/heart/robustness.csv`:

| Model | Clean videos | Webcam-like (degraded) | Resting (~70 BPM) |
|---|---|---|---|
| POS (baseline) | 1.29 | - | - |
| PhysNet v1 | 1.83 | 3.24 | 6.29 |
| PhysNet v2 (webcam augmentation) | 5.43 | 5.88 | 8.66 |
| **PhysNet v3** (frequency + webcam augmentation) | 1.88 | **2.67** | **5.14** |

What we learned on the way, measured step by step:
1. **Webcam auto-exposure** changed the skin brightness in steps and created fake pulses. The Heart
   Rate page now fixes the exposure and white balance (brightness stable, 20 → 30 fps).
2. **Domain shift**: live against a smartwatch, POS was 3.2 BPM off and PhysNet v1 10.4 (always too
   high). Webcam-like augmentation (JPEG, noise, blur, irregular frame rate) gave v2.
3. **Label bias**: per-subject analysis showed that the only failures come from the one test subject
   with a slow pulse (59 BPM, read as 2×). UBFC subjects play a stressful game, so their heart rates are
   mostly 85–120 BPM, and the model leans high. **Frequency augmentation** (video and PPG played slower or
   faster, heart rates 45–150 BPM) gave **v3**, the default model in the app.

Figures: `output/heart/v1|v2|v3/` (scatter, Bland-Altman, learning curves, example wave).
The app shows POS and PhysNet side by side and logs both against a smartwatch in
`output/heart/validation.csv`.

## 3. Stress (pilot, multimodal)

The planned dataset (UBFC-Phys) could not be downloaded in time, so we built a **pilot dataset with
the same protocol** in the app (*Stress* page):
- **rest**: sit and breathe slowly with the guide (2 min)
- **stress**: timed mental arithmetic, 3 choices, 4 s per question, beep when wrong or too slow (2 min)

Only numbers are saved (52 facial blendshapes + skin colour per frame), never video:
`data/stress/<person>/<rest|stress>_NN.npz`.

Model (`smartclass/models/stress_models.py`), on 20 s windows:
- face: 52 blendshapes × 200 steps (10 Hz) → **LSTM**
- pulse: POS wave, 600 samples (30 Hz) → **1D CNN**
- fusion: both embeddings → MLP → P(stress), shown live as a 0–100 score

```bash
python -m smartclass.training.train_stress      # or the "Train stress models" button
```

**Results** (one person, 3 rest + 3 stress sessions, 3-fold cross-validation **by session**):

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Face only (LSTM) | 0.79 ± 0.15 | 0.71 | 0.95 |
| Pulse only (1D CNN) | 0.63 ± 0.13 | 0.52 | 0.86 |
| **Fusion** | **0.91 ± 0.06** | **0.90** | **0.97** |

The fusion is better and more stable: in each fold where one modality fails, the other one carries the
decision. This is a **proof of concept** (one person; the model may also react to the task itself), and
the next step is more people or UBFC-Phys. Figures: `output/stress/`.

## Ethics

Participants opt in and are identified only by a pseudonym. No video is ever stored: only landmarks,
blendshape scores and colour traces. The classroom modules are meant to show class averages only, and
heart rate and stress are **indicators, not a medical diagnosis**. The datasets in `data/` contain only
the author's own data.

## Layout

```
smartclass/
├── camera.py, vision/        camera thread (fixed exposure for rPPG), MediaPipe hands/face (tools)
├── data/                     gesture dataset I/O, features shared by training and live inference
├── rppg/                     POS, UBFC preprocessing, live heart-rate processors (POS + PhysNet)
├── stress/                   stress sessions, 20 s window features, live stress processor
├── models/                   our PyTorch models: gesture MLP/LSTM/GRU/CNN, PhysNet, StressNet
├── training/                 datasets, augmentation, training loops, metrics, figures
├── inference/                live gesture recognition, decision logic, key sending
└── ui/                       PySide6 pages (dark theme)
notebooks/                    Colab notebooks (gestures, PhysNet)
output/                       figures and result tables for the report
```
