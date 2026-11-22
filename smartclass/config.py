"""Shared configuration: paths, model files, gestures."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

APP_NAME = "Smart Classroom"
APP_VERSION = "0.6 (Step 6)"

# Pretrained MediaPipe models (tools only: the deep-learning models of the project are trained by us)
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"

# Step 5: heart rate (rPPG)
HR_WINDOW_S = 10.0      # seconds of signal used for one BPM estimate
HR_MIN_S = 5.0          # first estimate after this many seconds
HR_FS = 30.0            # traces are resampled to this rate (the camera FPS is irregular)
HR_BAND = (0.7, 4.0)    # Hz = 42-240 BPM
HR_DIR = OUTPUT_DIR / "heart"

# Step 6: PhysNet on UBFC-rPPG
UBFC_CACHE = DATA_DIR / "ubfc_cache"   # preprocessed subjects (face crops + ROI RGB + PPG), one .npz each
PHYS_SIZE = 72          # face crop size (pixels)
PHYS_T = 128            # frames per training chunk (~4.3 s at 30 fps)
PHYS_FS = 30.0          # UBFC-rPPG frame rate; live frames are resampled to it

# Gestures for Magic Slides (Step 2 dataset, Step 3 models)
GESTURES = {
    "none": "rest / no gesture",
    "swipe_right": "next slide",
    "swipe_left": "previous slide",
    "point": "laser pointer",
    "pinch": "zoom",
    "open_palm": "black screen / pause",
}
# Recording protocol shown to every participant (keep it identical for everyone)
GESTURE_HELP = {
    "none": "Hand visible but idle, or moving naturally (scratching, talking). Also record with NO hand.",
    "swipe_right": "Open hand on the LEFT of the screen, sweep it to the RIGHT once, then HOLD it there "
                   "until REC ends. Don't bring it back.",
    "swipe_left": "Open hand on the RIGHT of the screen, sweep it to the LEFT once, then HOLD it there "
                  "until REC ends. Don't bring it back.",
    "point": "Index finger extended, others folded; move the fingertip slowly around.",
    "pinch": "Thumb and index start apart, then pinch together (or open from a pinch).",
    "open_palm": "Open palm facing the camera, fingers spread, hold still.",
}
SEQ_LEN = 30  # frames per gesture sample (~1-1.5 s depending on FPS)

# Step 2: gesture dataset  ->  data/gestures/<person>/<gesture>/<nnnn>.npz
GESTURE_DIR = DATA_DIR / "gestures"
SAMPLES_PER_GESTURE = 100   # target per person and gesture
CROP_SIZE = 96              # hand crops for the CNN models (stored as JPEG inside the .npz)
MIN_HAND_RATIO = 0.6        # a gesture sample is rejected if the hand is visible in < 60% of the frames

CAMERA_WIDTH, CAMERA_HEIGHT = 960, 540
