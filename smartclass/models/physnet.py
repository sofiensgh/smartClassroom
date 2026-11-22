"""PhysNet (3D CNN) - our PyTorch implementation of

Z. Yu, X. Li, G. Zhao, "Remote Photoplethysmograph Signal Measurement from Facial Videos
Using Spatio-Temporal Networks", BMVC 2019 (PhysNet-3DCNN-ED).

Input  (B, 3, T, H, W) face-crop video, T divisible by 4
Output (B, T)          the rPPG signal (one value per frame)
Encoder: 3D convolutions + pooling (space 4x smaller x2, time /4); decoder: 2 transposed
convolutions bring time back to T; then global spatial pooling and a 1x1x1 conv.
"""
from torch import nn


def _block(cin, cout, k, p):
    return nn.Sequential(nn.Conv3d(cin, cout, k, padding=p, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))


class PhysNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        c1, c2 = width // 2, width   # original: 16 / 32 / 64 channels; width=64 gives the paper size
        c3 = width * 2
        self.encoder = nn.Sequential(
            _block(3, c1, (1, 5, 5), (0, 2, 2)),
            nn.MaxPool3d((1, 2, 2)),
            _block(c1, c2, 3, 1), _block(c2, c3, 3, 1),
            nn.MaxPool3d((2, 2, 2)),
            _block(c3, c3, 3, 1), _block(c3, c3, 3, 1),
            nn.MaxPool3d((2, 2, 2)),
            _block(c3, c3, 3, 1), _block(c3, c3, 3, 1),
            nn.MaxPool3d((1, 2, 2)),
            _block(c3, c3, 3, 1), _block(c3, c3, 3, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(c3, c3, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)), nn.BatchNorm3d(c3), nn.ELU(),
            nn.ConvTranspose3d(c3, c3, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)), nn.BatchNorm3d(c3), nn.ELU(),
        )
        self.head = nn.Conv3d(c3, 1, 1)

    def forward(self, x):
        t = x.shape[2]
        z = self.decoder(self.encoder(x))                 # (B, C, T, h, w)
        z = nn.functional.adaptive_avg_pool3d(z, (t, 1, 1))
        return self.head(z).flatten(1)                    # (B, T)


def neg_pearson_loss(pred, target, eps=1e-8):
    """1 - Pearson correlation, averaged over the batch: only the SHAPE of the pulse matters,
    not its amplitude or offset (which depend on skin tone and light)."""
    p = pred - pred.mean(1, keepdim=True)
    t = target - target.mean(1, keepdim=True)
    r = (p * t).sum(1) / ((p.norm(dim=1) * t.norm(dim=1)) + eps)
    return (1 - r).mean()
