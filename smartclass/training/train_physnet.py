"""Step 6 - train PhysNet on UBFC-rPPG and compare it with POS on the SAME test subjects and windows.

    python -m smartclass.training.train_physnet --cache data/ubfc_cache --epochs 30

Outputs
    models/physnet.pt                       best checkpoint (lowest validation MAE)
    output/heart/physnet_curves.png         training loss + validation MAE per epoch
    output/heart/results.csv                MAE / RMSE / Pearson r of POS vs PhysNet (test subjects)
    output/heart/windows.csv                every 10 s test window: reference, POS and PhysNet BPM
    output/heart/scatter.png                estimated vs reference BPM, both methods
    output/heart/bland_altman.png           agreement plot (PhysNet and POS)
    output/heart/example_wave.png           reference PPG vs PhysNet vs POS on a test subject
    output/heart/split.txt                  subjects in train / val / test
"""
import argparse
import copy
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from ..config import HR_DIR, MODELS_DIR, PHYS_SIZE, PHYS_T, UBFC_CACHE
from ..models.physnet import PhysNet, neg_pearson_loss
from ..rppg.pos import bandpass, pos, spectrum_bpm
from .rppg_data import ChunkDataset, load_cache, predict_signal, split_subjects, zscore

WIN_S, STEP_S = 10.0, 2.0   # HR is evaluated on 10 s windows every 2 s (same protocol for both methods)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def window_hrs(sig_by_method, fs):
    """{method: signal} -> list of {method: BPM} for every 10 s window (reference = 'ref')."""
    n = len(next(iter(sig_by_method.values())))
    w, s = int(WIN_S * fs), int(STEP_S * fs)
    rows = []
    for i in range(0, n - w + 1, s):
        rows.append({m: spectrum_bpm(bandpass(sig[i:i + w], fs), fs)[0] for m, sig in sig_by_method.items()})
    return rows


def evaluate(model, data, subjects, device, with_pos=False):
    rows = []
    for sid in subjects:
        d = data[sid]
        fs = float(d["fps"])
        sigs = {"ref": d["ppg"].astype(np.float64), "PhysNet": predict_signal(model, d["frames"], device)}
        if with_pos:
            sigs["POS"] = pos(d["rgb"], fs)
        rows += [{"subject": sid, **r} for r in window_hrs(sigs, fs)]
    return rows


def metrics(rows, method):
    ref = np.array([r["ref"] for r in rows])
    est = np.array([r[method] for r in rows])
    err = est - ref
    r = float(np.corrcoef(ref, est)[0, 1]) if len(rows) > 2 and est.std() > 0 else float("nan")
    return {"method": method, "MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "pearson_r": r, "windows": len(rows)}


def _fig(path, draw, size=(6, 5)):
    fig = Figure(figsize=size)
    FigureCanvasAgg(fig)
    draw(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def save_figures(rows, history, example, out):
    ref = np.array([r["ref"] for r in rows])

    def curves(fig):
        a1, a2 = fig.subplots(1, 2)
        ep = [h["epoch"] for h in history]
        a1.plot(ep, [h["train_loss"] for h in history])
        a1.set(xlabel="epoch", ylabel="1 - Pearson r", title="Training loss")
        a2.plot(ep, [h["val_mae"] for h in history], color="tab:orange")
        a2.set(xlabel="epoch", ylabel="MAE (BPM)", title="Validation MAE")
        for a in (a1, a2):
            a.grid(alpha=0.3)
    _fig(out / "physnet_curves.png", curves, (10, 4))

    def scatter(fig):
        ax = fig.subplots()
        for m, c in (("POS", "tab:gray"), ("PhysNet", "tab:red")):
            ax.scatter(ref, [r[m] for r in rows], s=10, alpha=0.6, c=c, label=m)
        lo, hi = ref.min() - 10, ref.max() + 10
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(xlabel="reference HR (BPM)", ylabel="estimated HR (BPM)", title="Test subjects, 10 s windows")
        ax.legend()
        ax.grid(alpha=0.3)
    _fig(out / "scatter.png", scatter)

    def bland(fig):
        axes = fig.subplots(1, 2, sharey=True)
        for ax, m in zip(axes, ("PhysNet", "POS")):
            est = np.array([r[m] for r in rows])
            mean, diff = (est + ref) / 2, est - ref
            md, sd = diff.mean(), diff.std()
            ax.scatter(mean, diff, s=10, alpha=0.6)
            for v, ls in ((md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")):
                ax.axhline(v, color="k", ls=ls, lw=1)
            ax.set(xlabel="mean of estimate and reference (BPM)", title=f"{m}: bias {md:+.1f}, ±{1.96 * sd:.1f}")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("estimate - reference (BPM)")
    _fig(out / "bland_altman.png", bland, (11, 4))

    def wave(fig):
        ax = fig.subplots()
        t = np.arange(len(example["ref"])) / example["fs"]
        for k, c in (("ref", "k"), ("PhysNet", "tab:red"), ("POS", "tab:gray")):
            ax.plot(t, zscore(bandpass(example[k], example["fs"])), c, lw=1.2,
                    label={"ref": "reference PPG"}.get(k, k), alpha=0.9 if k != "POS" else 0.6)
        ax.set(xlabel="time (s)", title=f"{example['subject']}: pulse signals (band-passed, normalised)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    _fig(out / "example_wave.png", wave, (10, 3.5))


def run(cache=UBFC_CACHE, epochs=30, batch=8, lr=1e-3, patience=8, seed=42, width=32,
        out=HR_DIR, models_dir=MODELS_DIR, log=print):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_cache(cache)
    tr, va, te = split_subjects(list(data), seed)
    out, models_dir = Path(out), Path(models_dir)
    out.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    note = f"train ({len(tr)}): {tr}\nval ({len(va)}): {va}\ntest ({len(te)}): {te}"
    (out / "split.txt").write_text(note + "\n", encoding="utf-8")
    log(f"{len(data)} subjects on {device}\n{note}")

    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(ChunkDataset(data, tr, augment=True, seed=seed), batch_size=batch, shuffle=True,
                          generator=g, num_workers=2 if device.type == "cuda" else 0, drop_last=True)
    model = PhysNet(width).to(device)
    log(f"PhysNet: {sum(p.numel() for p in model.parameters()):,} parameters, {len(train_dl)} batches/epoch")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    history, best, best_state, bad = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                pred = model(x)
            loss = neg_pearson_loss(pred.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        val_mae = metrics(evaluate(model, data, va, device), "PhysNet")["MAE"]
        sched.step(val_mae)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": val_mae})
        log(f"epoch {epoch}/{epochs}  loss {np.mean(losses):.3f}  val MAE {val_mae:.2f} BPM  ({time.time() - t0:.0f}s)")
        if val_mae < best:
            best, best_state, bad = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience and epoch >= 10:
                log(f"early stopping (best val MAE {best:.2f})")
                break

    model.load_state_dict(best_state)
    torch.save({"state_dict": model.cpu().state_dict(), "width": width, "t": PHYS_T, "size": PHYS_SIZE,
                "fs": 30.0, "val_mae": best}, models_dir / "physnet.pt")
    model.to(device)

    rows = evaluate(model, data, te, device, with_pos=True)
    results = [metrics(rows, "POS"), metrics(rows, "PhysNet")]
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in results])
    with open(out / "windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "ref", "POS", "PhysNet"])
        w.writeheader()
        w.writerows([{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    d = data[te[0]]
    n = int(20 * float(d["fps"]))
    example = {"subject": te[0], "fs": float(d["fps"]), "ref": d["ppg"][:n].astype(np.float64),
               "PhysNet": predict_signal(model, d["frames"][:max(n, PHYS_T)], device)[:n], "POS": pos(d["rgb"][:n], float(d["fps"]))}
    save_figures(rows, history, example, out)
    for r in results:
        log(f"TEST {r['method']:8s} MAE {r['MAE']:.2f}  RMSE {r['RMSE']:.2f}  r {r['pearson_r']:.3f}  ({r['windows']} windows)")
    log(f"Saved models/physnet.pt and figures in {out}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=UBFC_CACHE)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32, help="32 = light (default), 64 = original paper size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=HR_DIR)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    a = ap.parse_args()
    run(a.cache, a.epochs, a.batch, a.lr, seed=a.seed, width=a.width, out=a.out, models_dir=a.models_dir)


if __name__ == "__main__":
    main()
