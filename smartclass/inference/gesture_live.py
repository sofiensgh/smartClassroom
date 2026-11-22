"""Step 4 - live gesture recognition with our trained model.

Runs INSIDE the CameraWorker thread (imports torch, never in the UI thread).
Keeps a sliding window of the last SEQ_LEN frames and builds exactly the same
features as in training (smartclass.data.gesture_features).
"""
import numpy as np
import torch

from ..config import MODELS_DIR
from ..data.gesture_features import mirror_x, sequence_features
from ..models.gesture_models import build_model


class GestureRecognizer:
    def __init__(self, name, models_dir=MODELS_DIR):
        torch.set_num_threads(2)   # small model: leave the other cores to MediaPipe and the UI
        ck = torch.load(models_dir / f"gesture_{name}.pt", map_location="cpu", weights_only=False)
        if ck["kind"] != "seq":
            raise ValueError(f"{name} needs image crops: not usable live")
        self.name, self.classes, self.motion = name, list(ck["classes"]), ck["motion"]
        self.model = build_model(ck["arch"], ck["in_dim"], len(self.classes))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        n = ck["seq_len"]
        self.L = np.zeros((n, 21, 3), np.float32)
        self.P = np.zeros(n, bool)
        self._hand = None
        self._reset = False

    def request_reset(self):
        """Clear the window (called from the UI thread after an action; read in the camera thread)."""
        self._reset = True

    def _pick(self, hands):
        if not hands:
            return None
        same = [h for h in hands if h.handedness == self._hand]
        h = max(same or hands, key=lambda h: h.score)
        self._hand = h.handedness
        return h

    def __call__(self, hands, mirror):
        if self._reset:
            self.L[:] = 0
            self.P[:] = False
            self._reset = False
        hand = self._pick(hands)
        self.L = np.roll(self.L, -1, axis=0)
        self.P = np.roll(self.P, -1)
        self.L[-1] = hand.landmarks if hand is not None else 0
        self.P[-1] = hand is not None
        L = self.L if mirror else mirror_x(self.L, self.P)   # the models were trained on the mirrored view
        x = torch.from_numpy(sequence_features(L, self.P, self.motion))[None]
        with torch.inference_mode():
            probs = torch.softmax(self.model(x), dim=1)[0].numpy()
        tip = None
        if hand is not None:
            tx, ty = float(hand.landmarks[8, 0]), float(hand.landmarks[8, 1])   # index fingertip
            tip = (tx if mirror else 1.0 - tx, ty)
        return {"probs": probs, "classes": self.classes, "tip": tip, "model": self.name}
