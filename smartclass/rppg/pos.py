"""Step 5 - classic rPPG baseline: POS (Plane-Orthogonal-to-Skin).

W. Wang, A. C. den Brinker, S. Stuijk, G. de Haan, "Algorithmic Principles of Remote PPG",
IEEE Trans. Biomed. Eng. 64(7), 2017.

No learning at all: this is the NON-deep baseline that PhysNet (Step 6) must beat.
Pipeline: mean skin RGB per frame -> resample to a fixed rate -> POS -> band-pass -> spectrum peak = BPM.
"""
from collections import deque

import numpy as np
from scipy.signal import butter, filtfilt

from ..config import HR_BAND, HR_FS, HR_MIN_S, HR_WINDOW_S

POS_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


def pos(rgb, fs, win_s=1.6):
    """(N, 3) RGB traces -> (N,) pulse signal (overlap-add of short POS windows)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    n, l = len(rgb), int(round(win_s * fs))
    out = np.zeros(n)
    if n < l:
        return out
    for m in range(n - l + 1):
        c = rgb[m:m + l]
        cn = c / np.maximum(c.mean(0), 1e-9)          # temporal normalisation (removes skin tone / light level)
        s = cn @ POS_PROJECTION.T                     # project on the plane orthogonal to the skin tone
        sd1 = s[:, 1].std()
        h = s[:, 0] + (s[:, 0].std() / sd1 if sd1 > 1e-12 else 0.0) * s[:, 1]   # alpha tuning
        out[m:m + l] += h - h.mean()                  # overlap-add
    return out


def bandpass(x, fs, band=HR_BAND, order=3):
    b, a = butter(order, band, btype="band", fs=fs)
    return filtfilt(b, a, x)


def spectrum_bpm(x, fs, band=HR_BAND, n_fft=4096):
    """Peak of the power spectrum inside the band -> (bpm, snr_db, freqs_bpm, power)."""
    x = (x - x.mean()) * np.hanning(len(x))
    n_fft = max(n_fft, len(x))
    power = np.abs(np.fft.rfft(x, n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    inb = (freqs >= band[0]) & (freqs <= band[1])
    f, p = freqs[inb], power[inb]
    f0 = f[np.argmax(p)]
    # SNR (de Haan & Jeanne 2013): power around the peak and its 1st harmonic vs. the rest of the band
    sig = (np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.2)
    snr = 10 * np.log10(p[sig].sum() / max(p[~sig].sum(), 1e-12))
    return f0 * 60.0, float(snr), f * 60.0, p / max(p.max(), 1e-12)


class PulseEstimator:
    """Accumulates (time, mean RGB) samples and estimates the heart rate on a sliding window."""

    def __init__(self, window_s=HR_WINDOW_S, min_s=HR_MIN_S, fs=HR_FS, max_gap_s=1.0):
        self.window_s, self.min_s, self.fs, self.max_gap_s = window_s, min_s, fs, max_gap_s
        self.t = deque()
        self.rgb = deque()
        self.history = deque(maxlen=5)   # recent good estimates -> median = displayed BPM

    def reset(self):
        self.t.clear()
        self.rgb.clear()
        self.history.clear()

    @property
    def span(self):
        return self.t[-1] - self.t[0] if len(self.t) > 1 else 0.0

    def push(self, t, rgb):
        """Add one frame. rgb=None (no face): a gap longer than max_gap_s restarts the measurement."""
        if rgb is None:
            if self.t and t - self.t[-1] > self.max_gap_s:
                self.reset()
            return
        if self.t and t - self.t[-1] > self.max_gap_s:
            self.reset()
        self.t.append(t)
        self.rgb.append(rgb)
        while self.t and t - self.t[0] > self.window_s:
            self.t.popleft()
            self.rgb.popleft()

    def estimate(self, min_snr_db=-3.0):
        """-> dict(bpm, bpm_raw, snr, wave, spec_bpm, spec) or None while there is not enough signal."""
        if self.span < self.min_s:
            return None
        t = np.asarray(self.t)
        rgb = np.asarray(self.rgb)
        tu = np.arange(t[0], t[-1], 1 / self.fs)                  # uniform time grid
        rgb_u = np.stack([np.interp(tu, t, rgb[:, c]) for c in range(3)], 1)
        wave = bandpass(pos(rgb_u, self.fs), self.fs)
        bpm, snr, fb, p = spectrum_bpm(wave, self.fs)
        if snr >= min_snr_db:
            self.history.append(bpm)
        shown = float(np.median(self.history)) if self.history else None
        tail = wave[-int(6 * self.fs):]                           # last 6 s for the live plot
        tail = tail / max(np.abs(tail).max(), 1e-12)
        return {"bpm": shown, "bpm_raw": bpm, "snr": snr, "wave": tail, "spec_bpm": fb, "spec": p}
