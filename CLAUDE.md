# Smart Classroom: project context for Claude

## Who / what
- Student project for the **NVIDIA DLI deep-learning course** (GLSI/SSIR, Tunisia). Course material is in the parent folder `../`: CNN = Chapter 2, RNN = Chapter 3, autoencoders = Chapter 4, plus U-Nets, diffusion, BERT, RAPIDS and Jetson Nano.
- **Smart Classroom** is a **desktop application** (PySide6/Qt) with three modules:
  1. **Magic Slides**: the teacher controls slides (PowerPoint, PDF) with hand gestures.
  2. **Heart Rate**: heart rate measured from a normal webcam (rPPG).
  3. **Stress Level**: an anonymous class-level stress indicator during exams, from the pulse signal and facial cues.
  4. A **teacher dashboard** combining them.
- The user writes in English, French or **Tunisian derja** (Latin letters, e.g. "behi", "9olli", "3lech"). Reply in the user's language and style.
- **Working preference:** build **step by step**. Finish one step, explain how to test it, then wait for "go / abda step X". Explain before changing things; don't do large, unrequested changes.

## 🧠 RULE #1: this must stay a DEEP-LEARNING project
The professor must see deep learning as the core, not "MediaPipe + scripts". Every change must respect these guardrails:

1. **Three deep models trained by the student (from scratch or fine-tuned), not pretrained black boxes:**
   | Model | Architecture | Data | Course chapter |
   |---|---|---|---|
   | Gesture recognizer | **LSTM/GRU over landmark sequences** (main) + **CNN on hand images / CNN+LSTM on clips** (comparison) | **Own recorded dataset** (Step 2) | Ch. 2 + 3 |
   | Heart-rate estimator | **PhysNet (3D CNN)**; optionally PhysFormer/Transformer | UBFC-rPPG | Ch. 2 |
   | Stress estimator | **Multimodal network**: CNN (face crops or blendshape sequence) + 1D-CNN/LSTM (pulse/HRV signal) → late fusion | UBFC-Phys | Ch. 2 + 3 |
2. **MediaPipe, OpenCV, pyautogui, FFT and filters are TOOLS only.** MediaPipe only turns pixels into keypoints, and gestures are NEVER decided by hand-written rules in the final app (rules are allowed only as a debug view, like the finger counter in Step 1).
3. **Every deep model has a non-deep or simpler baseline and a comparison:**
   - gestures: MLP on a single frame (and/or rule-based) vs LSTM vs CNN
   - heart rate: classic **POS** algorithm vs PhysNet (MAE / RMSE in BPM, Pearson r)
   - stress: single-modality (face only, signal only) vs multimodal fusion
4. **Proper ML methodology:** train/val/test splits **by person/subject** (no leakage between the same person's samples), augmentation, early stopping or best checkpoint, fixed seeds, learning curves, **confusion matrix**, per-class metrics, and an ablation where possible.
5. **Explainability:** Grad-CAM / saliency for the CNNs, attention or feature importance where possible. Save the figures to `output/` for the report.
6. **Training code is our own PyTorch:** Dataset classes, model classes, training loops, metrics. No AutoML and no "pip install a finished rPPG library" as the main model (such libraries may serve as references or baselines only).
7. **Contribution:** the student's own gesture dataset (several people, documented protocol) + the deep-vs-classic comparison.
8. **Ethics section** (built into the UI home page): opt-in consent, class-average only (never a named student), no video stored, "indicator, not medical diagnosis".
9. Deliverables for the report are generated automatically: `output/*.png` curves, confusion matrices, comparison tables (CSV), Grad-CAM images.

## Environment
- Windows 11, PowerShell. System Python 3.14.
- **Virtual env lives OUTSIDE OneDrive** (to avoid syncing thousands of files): `C:\Users\sofie\.venvs\smartclassroom`
  - Run: `C:\Users\sofie\.venvs\smartclassroom\Scripts\python.exe main.py` from the `SmartClassroom` folder.
  - **Install status (2026-11-22): DONE.** torch 2.15.0+cpu, PySide6-Essentials 6.12, opencv-contrib 5.1, mediapipe 1.2.3, scipy, pandas, matplotlib, pyautogui. Check: `python -c "import PySide6, cv2, mediapipe, scipy; print('JAHEZ')"`.
    - Use **PySide6-Essentials**, not full PySide6 (that adds ~200 MB of WebEngine; the network is slow, ~500 KB/s, so pip looked "stuck").
    - Use **opencv-contrib-python** only (mediapipe depends on it); adding opencv-python as well gives two clashing `cv2` packages.
    - No GPU locally; heavy training goes to a **Kaggle notebook with a T4 GPU**.
- **MediaPipe ≥ 1.x has NO `mp.solutions`.** Use the Tasks API: `from mediapipe.tasks.python import vision, BaseOptions`, `vision.HandLandmarker` / `vision.FaceLandmarker`, `RunningMode.VIDEO`, `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)`, with timestamps that strictly increase.
- Model files download automatically on first use to `models/` (`hand_landmarker.task` from storage.googleapis.com, via `smartclass/vision/assets.py`).
- A sister project `EmoMusic` (emotion CNN + LSTM music, Face Orchestra using MediaPipe FaceLandmarker blendshapes + a real-time synth) exists in the user's Downloads copy. Its face-tracking code can be reused for the face/rPPG part.

## Code layout (current)
```
SmartClassroom/
├── main.py                      launches the Qt app (Fusion style + dark theme)
├── requirements.txt             torch, torchvision, PySide6, opencv-python, mediapipe, numpy, scipy, pandas, matplotlib, pyautogui
├── data/  models/  output/      (created as needed)
└── smartclass/
    ├── config.py                paths, APP_VERSION, HAND_MODEL_URL/PATH, GESTURES dict, SEQ_LEN=30, camera resolution
    ├── camera.py                CameraWorker(QThread): DirectShow capture, mirror, HandTracker, optional hand crops,
    │                            emits frame_ready(rgb, {fps, hands, crops, mirror}); stop() is race-safe
    ├── data/
    │   └── gesture_dataset.py   save_sample / load_sample / count_samples / list_samples
    │                            data/gestures/<person>/<gesture>/<nnnn>.npz: RAW landmarks (T,21,3) + present + handedness
    │                            + crops as JPEG bytes (~18 KB/sample). Raw landmarks keep the MOTION: features() is
    │                            wrist-centred (shape only), so Step 3 must add the wrist trajectory for swipes.
    ├── vision/
    │   ├── assets.py            ensure_file(url, path, status), atomic download
    │   └── hands.py             HandTracker (MediaPipe VIDEO), Hand dataclass (landmarks 21x3, handedness, score,
    │                            fingers_up(), features() → 63-d wrist-centred, scale-normalised vector), draw_hands()
    ├── data/gesture_features.py sequence_features(L, P, motion): 68-d per frame = 63 shape + 2 wrist pos + 2 wrist vel
    │                            + present flag (64-d without motion = ablation). SHARED by training and Step 4 live inference.
    ├── models/gesture_models.py FrameMLP (baseline), GestureRNN (lstm/gru, unidirectional for live use), CNNGRU (crops)
    ├── training/
    │   ├── specs.py             MODEL_SPECS: mlp, lstm, gru, lstm_shape (ablation), cnn (no torch import: UI lists them)
    │   ├── gesture_data.py      load_all (mirror-normalised), split_by_person (random fallback <3 people, with warning),
    │   │                        augmentation (mirror + swipe label swap, rotate/scale/shift/jitter, speed), Datasets
    │   ├── train_gestures.py    run(cfg, log, progress, should_stop) + CLI `python -m smartclass.training.train_gestures`
    │   │                        AdamW, class-weighted CE + label smoothing, ReduceLROnPlateau, early stop on val macro-F1
    │   │                        -> models/gesture_<name>.pt (with classes/seq_len/motion meta) + output/gestures/*
    │   ├── metrics.py / plots.py  numpy metrics; matplotlib Figure+Agg (thread-safe, no pyplot)
    │   └── worker.py            TrainWorker(QThread): log/progress/done/failed signals, stop after current batch
    ├── inference/
    │   ├── controller.py        (no torch) ACTIONS map, available_models(), model_scores(), GestureController:
    │   │                        EMA + threshold 0.8 + 3 consecutive frames + hold (open_palm 0.6 s) + cooldown 1.2 s
    │   │                        + "release before repeat"; pointing with hysteresis
    │   └── gesture_live.py      GestureRecognizer: sliding window SEQ_LEN, same features as training, runs in the
    │                            CameraWorker thread (CameraWorker.set_processor_factory builds it there: torch never
    │                            imported in the UI thread). request_reset() after each action.
    │                            Replay of p01 data (15 idle + 30 + 15 hold frames): LSTM 97/99 right, 0 double fires;
    │                            MLP 6 double fires -> report point "temporal model is more stable live".
    └── ui/
        ├── overlay.py           PointerOverlay: frameless, click-through, always-on-top laser dot + action toast.
        │                        Must never be (re)shown while closing, or it keeps the app alive (bug fixed).
        ├── theme.py             dark QSS (object names: Sidebar, Title, Subtitle, Card, CardTitle, Muted, Badge, BadgeDone, Video, BigValue, Primary, Danger)
        ├── widgets.py           VideoView (numpy → QPixmap), card(), label(), PlaceholderPage
        ├── home_page.py         overview, 3 DL model cards, ROADMAP list (update the done flags as steps finish), ethics card
        ├── camera_page.py       CameraPage base: camera controls + worker lifecycle; hooks on_frame(), camera_stopped(); add_side()
        ├── slides_page.py       Step 1 page (CameraPage): FPS, hands + fingers info (debug view)
        ├── training_page.py     Step 2 recorder (CameraPage): person pseudonym, gesture, countdown, REC overlay,
        │                        Record 1 / Auto to 100 / Stop / Delete last, live counts table. Step 3 training goes here too.
        └── main_window.py       sidebar + QStackedWidget; switching page stops the other page's camera; close stops all
```
- Status: Steps 0–1 **launch fine** (window opens, hand tracker + camera 0 tested headlessly, 2026-11-22). Bugs fixed: stop-during-startup race in CameraWorker, VideoView growing loop, button restyle. Still to check by hand with a real hand in front of the camera: Test first: the app opens, the pages switch, the camera starts and stops, hands are drawn, and closing releases the camera.

## Roadmap (do one step at a time)
| Step | Content | Deep learning part |
|---|---|---|
| 0 ✅ (written) | Desktop skeleton, env | - |
| 1 ✅ (written) | Live hand tracking page | tool only |
| 2 ✅ (written + headless-tested 2026-11-22; user must record real data) | **Gesture dataset recorder** in "Data & Training": choose gesture + person ID, countdown, record SEQ_LEN=30 frames of `features()` (+ optional 96×96 hand crops for the CNN), save `data/gestures/<person>/<gesture>/<n>.npz`, live counters, ~100 samples/gesture, several people | builds our dataset |
| 3 ✅ code + first real run (2026-11-22, Colab T4): p01 only, 99 samples, RANDOM split (1 person) -> test (19 samples) MLP 1.00, LSTM 0.95, GRU 0.84, ablation 0.95, CNN 0.84: pipeline proof only, not report numbers. Results in models/ + output/gestures. Fixes learned: tiny data needs small batches (auto batch = n_train//8) + min_epochs=30, patience 20; swipes must not "return" inside the 2 s window (15 FPS); one zero-filled .npz in OneDrive -> atomic verified save. NEXT: record p02, p03+ then retrain | **Train gesture models** (also launchable from the UI with a progress bar via QThread): MLP baseline, LSTM/GRU main, CNN or CNN+LSTM on crops. Split by person. Curves, confusion matrices, results table | ✅ core DL |
| 4 ✅ (2026-11-22, WORKS with PowerPoint, confirmed by the user). Key lesson: keys go to the focused window (our app after ticking "Send keys") -> inference/keys.py focuses the slideshow window (class "screenClass", Alt tap + SetForegroundWindow) before pyautogui.press. PostMessage WM_KEYDOWN does NOT work with PowerPoint. COM (comtypes) would be more robust but needs the user's OK to install | **Magic Slides live**: LSTM on a sliding window → smoothed prediction + confidence threshold + cooldown → pyautogui keys (→ ← B), laser-pointer overlay (transparent always-on-top Qt window), zoom | uses the trained model |
| 5 | **Heart rate, classic POS baseline**: FaceLandmarker ROIs (forehead + cheeks), RGB traces, POS, band-pass 0.7–4 Hz, FFT → BPM; live pulse plot; validate against a smartwatch/oximeter | baseline |
| 6 | **PhysNet** (3D CNN) trained on UBFC-rPPG (Kaggle GPU), negative Pearson loss; compare MAE/RMSE/r with POS; live inference in the app | ✅ core DL |
| 7 | **Stress multimodal model** on UBFC-Phys: face branch + signal branch + fusion; compare with single modalities; output 0–100 + relaxed/normal/stressed | ✅ core DL |
| 8 | **Teacher dashboard**: gesture status, anonymous class-average stress over time, alerts; students send numbers only | - |
| 9 | Evaluation + report figures, Grad-CAM, demo video; optional PyInstaller `.exe` | explainability |

## Conventions
- Heavy work (camera, inference, training) runs in a **QThread**, never in the UI thread. Talk to the UI only through Signals.
- Keep the dark theme and the `card()` / `label()` helpers for a consistent look.
- Put new pages in `smartclass/ui/`, models in `smartclass/models/`, training in `smartclass/training/`, data code in `smartclass/data/`.
- After each step: update `APP_VERSION`, the ROADMAP flags in `home_page.py`, and this file's status.
- Explain to the user (in their language) how to test each step.
