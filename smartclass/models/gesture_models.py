"""Gesture recognition models (Step 3), written from scratch in PyTorch.

    FrameMLP   baseline: classifies every frame on its own, then averages -> no temporal modelling
    GestureRNN main:     LSTM or GRU over the landmark sequence (course Ch. 3)
    CNNGRU     image:    small CNN on each hand crop + GRU over the clip (course Ch. 2 + 3)
"""
import torch
from torch import nn


class FrameMLP(nn.Module):
    """Per-frame MLP; the sample prediction is the mean of the frame logits over frames with a hand."""

    def __init__(self, in_dim, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes))

    def forward(self, x):                      # x: (B, T, F), last feature = "hand present" flag
        logits = self.net(x)                   # (B, T, C)
        w = x[..., -1:]
        w = torch.where(w.sum(1, keepdim=True) > 0, w, torch.ones_like(w))  # no hand at all -> plain mean
        return (logits * w).sum(1) / w.sum(1)


class GestureRNN(nn.Module):
    """Linear projection -> 2-layer LSTM/GRU -> classifier on the last time step.

    Unidirectional on purpose: in Step 4 it runs live on a sliding window of past frames.
    """

    def __init__(self, in_dim, n_classes, hidden=128, layers=2, cell="lstm", dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):                      # (B, T, F)
        h, _ = self.rnn(self.proj(x))
        return self.head(h[:, -1])


def _conv_block(cin, cout, pool=True):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class CNNGRU(nn.Module):
    """CNN encoder applied to every hand crop, then a GRU over the clip.

    `self.cnn[-1]` is the last conv block: the Grad-CAM target in Step 9.
    """

    def __init__(self, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            _conv_block(3, 32), _conv_block(32, 64), _conv_block(64, 128), _conv_block(128, 128, pool=False))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.rnn = nn.GRU(128, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def encode(self, x):                       # (B, T, 3, H, W) -> (B, T, 128)
        b, t = x.shape[:2]
        f = self.pool(self.cnn(x.flatten(0, 1))).flatten(1)
        return f.view(b, t, -1)

    def forward(self, x):
        h, _ = self.rnn(self.encode(x))
        return self.head(h[:, -1])


def build_model(arch, in_dim, n_classes):
    if arch == "mlp":
        return FrameMLP(in_dim, n_classes)
    if arch in ("lstm", "gru"):
        return GestureRNN(in_dim, n_classes, cell=arch)
    if arch == "cnn_gru":
        return CNNGRU(n_classes)
    raise ValueError(f"unknown architecture {arch!r}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
