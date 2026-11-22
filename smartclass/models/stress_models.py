"""Step 7 - multimodal stress network (our PyTorch).

    face  (B, 200, 52) blendshape sequence --LSTM(64)--------> 64-d  \
                                                                        concat -> MLP -> P(stress)
    pulse (B, 600)     POS pulse wave     --1D CNN (3 conv)--> 64-d  /

mode = "face" | "pulse" (single-modality BASELINES) | "fusion" (the multimodal model).
"""
import torch
from torch import nn


class FaceBranch(nn.Module):
    """Facial expressions over time (Ch. 3, RNN)."""

    def __init__(self, n_in=52, hidden=64):
        super().__init__()
        self.drop = nn.Dropout(0.2)
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.out_dim = hidden

    def forward(self, x):
        _, (h, _) = self.lstm(self.drop(x))
        return h[-1]


class PulseBranch(nn.Module):
    """Pulse wave shape and rhythm (Ch. 2, CNN in 1D)."""

    def __init__(self, ch=(16, 32, 64)):
        super().__init__()
        layers, cin = [], 1
        for c in ch:
            layers += [nn.Conv1d(cin, c, 7, stride=2, padding=3), nn.BatchNorm1d(c), nn.ReLU(inplace=True)]
            cin = c
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.out_dim = ch[-1]

    def forward(self, x):
        return self.net(x.unsqueeze(1)).flatten(1)


class StressNet(nn.Module):
    def __init__(self, mode="fusion"):
        super().__init__()
        self.mode = mode
        self.face = FaceBranch() if mode in ("face", "fusion") else None
        self.pulse = PulseBranch() if mode in ("pulse", "fusion") else None
        d = (self.face.out_dim if self.face else 0) + (self.pulse.out_dim if self.pulse else 0)
        self.head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, face, pulse):
        z = [b(x) for b, x in ((self.face, face), (self.pulse, pulse)) if b is not None]
        return self.head(torch.cat(z, 1)).squeeze(1)   # logit of "stress"
