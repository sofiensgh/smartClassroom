"""Step 4 - turn the model's per-frame probabilities into slide actions (no torch here).

The GESTURE is always decided by the trained network. This module only makes the
output usable live: smoothing (EMA), a confidence threshold, N consecutive frames,
a minimum hold time for static gestures, a cooldown, and "release before repeat".
"""
import csv

import numpy as np

from ..config import MODELS_DIR, OUTPUT_DIR
from ..training.specs import MODEL_SPECS

# gesture -> keys sent (pinch alternates between its keys) and the text shown to the user
ACTIONS = {
    "swipe_right": {"keys": ["right"], "label": "▶  Next slide"},
    "swipe_left": {"keys": ["left"], "label": "◀  Previous slide"},
    "open_palm": {"keys": ["b"], "label": "■  Black screen on/off"},
    "pinch": {"keys": ["+", "-"], "label": "🔍  Zoom"},
}
HOLD_S = {"open_palm": 0.6, "pinch": 0.3}   # static gestures must be held a little (no accidental trigger)
PREFERRED = ["lstm", "gru", "lstm_shape", "mlp"]


def available_models(models_dir=MODELS_DIR):
    """Trained landmark models usable live (the CNN needs image crops, so it is not listed)."""
    names = [n for n, s in MODEL_SPECS.items() if s["kind"] == "seq" and (models_dir / f"gesture_{n}.pt").exists()]
    return sorted(names, key=lambda n: PREFERRED.index(n) if n in PREFERRED else 99)


def model_scores(out_dir=OUTPUT_DIR / "gestures"):
    """{model: test accuracy} from the last training run, for display."""
    path = out_dir / "results.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["model"]: float(r["test_acc"]) for r in csv.DictReader(f)}


class GestureController:
    def __init__(self, classes, threshold=0.8, cooldown=1.2, consecutive=3, ema=0.5):
        self.classes = list(classes)
        self.threshold, self.cooldown, self.consecutive, self.ema_w = threshold, cooldown, consecutive, ema
        self.reset()
        self._last_fire_t = -1e9

    def reset(self):
        """Forget the recent history (called after an action, together with the recognizer window)."""
        self.ema = None
        self.pointing = False
        self._streak_g, self._streak_n, self._streak_t0 = "none", 0, 0.0
        self._blocked = getattr(self, "_blocked", None)

    def update(self, probs, t):
        probs = np.asarray(probs, dtype=np.float32)
        self.ema = probs if self.ema is None else self.ema_w * self.ema + (1 - self.ema_w) * probs
        k = int(np.argmax(self.ema))
        gesture, conf = self.classes[k], float(self.ema[k])
        active = gesture if conf >= self.threshold and gesture != "none" else "none"

        # laser pointer: continuous state with hysteresis (no flicker around the threshold)
        if self.pointing:
            self.pointing = gesture == "point" and conf >= self.threshold - 0.15
        else:
            self.pointing = active == "point"

        if active == self._streak_g:
            self._streak_n += 1
        else:
            self._streak_g, self._streak_n, self._streak_t0 = active, 1, t
        if self._blocked and active != self._blocked:   # the gesture was released -> can fire again
            self._blocked = None

        event = None
        if (active in ACTIONS and active != self._blocked
                and self._streak_n >= self.consecutive
                and t - self._streak_t0 >= HOLD_S.get(active, 0.0)
                and t - self._last_fire_t >= self.cooldown):
            event, self._last_fire_t, self._blocked = active, t, active
        return {"gesture": gesture, "conf": conf, "active": active, "event": event, "pointing": self.pointing}
