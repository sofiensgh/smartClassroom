"""Step 3 - train and compare the gesture models on our own dataset.

Command line (also works on Kaggle after uploading the project + data/gestures):
    python -m smartclass.training.train_gestures                     # all models
    python -m smartclass.training.train_gestures --models lstm mlp --epochs 40

From the UI: Data & Training page -> "Train models" (runs this in a QThread).

Outputs
    models/gesture_<name>.pt               best checkpoint + everything needed for live inference (Step 4)
    output/gestures/<name>_curves.png      learning curves (loss, accuracy)
    output/gestures/<name>_confusion.png   confusion matrix on the TEST people
    output/gestures/<name>_per_class.csv   precision / recall / F1 per gesture
    output/gestures/results.csv            comparison table of all models
    output/gestures/comparison.png         bar chart of the comparison
    output/gestures/split.txt              which people are in train / val / test
"""
import argparse
import copy
import csv
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import GESTURE_DIR, MODELS_DIR, OUTPUT_DIR, SEQ_LEN
from ..data.gesture_features import feature_dim
from ..models.gesture_models import build_model, count_params
from . import metrics, plots
from .gesture_data import CLASSES, CLIP_FRAMES, CLIP_SIZE, ClipDataset, SequenceDataset, load_all, split_by_person
from .specs import MODEL_SPECS


class StopTraining(Exception):
    pass


@dataclass
class TrainConfig:
    models: list = field(default_factory=lambda: list(MODEL_SPECS))
    epochs: int = 80
    batch_size: int = 64        # upper bound: small datasets get smaller batches (>= ~8 steps per epoch)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20          # early stopping on validation macro-F1 ...
    min_epochs: int = 30        # ... but never before this (a tiny noisy val set plateaus early)
    seed: int = 42
    data_dir: Path = GESTURE_DIR
    out_dir: Path = OUTPUT_DIR / "gestures"
    models_dir: Path = MODELS_DIR


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, loss_fn, device, opt=None, should_stop=None):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum, preds, trues = 0, 0, 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            if should_stop and should_stop():
                raise StopTraining
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # keeps the RNNs stable
                opt.step()
            p = out.argmax(1)
            loss_sum += loss.item() * len(y)
            correct += (p == y).sum().item()
            total += len(y)
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    y_true, y_pred = np.concatenate(trues), np.concatenate(preds)
    return loss_sum / max(total, 1), correct / max(total, 1), y_true, y_pred


def _latency_ms(model, sample, device, runs=30):
    """Average time for ONE prediction (batch of 1), what matters for live use."""
    model.eval()
    x = sample.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        t = time.perf_counter()
        for _ in range(runs):
            model(x)
    return (time.perf_counter() - t) / runs * 1000


def train_one(name, data, split, cfg, device, log, progress, should_stop):
    spec = MODEL_SPECS[name]
    tr, va, te = split
    if spec["kind"] == "seq":
        make = lambda idx, aug, s: SequenceDataset(data, idx, augment=aug, motion=spec["motion"], seed=s)  # noqa: E731
        in_dim = feature_dim(spec["motion"])
    else:
        make = lambda idx, aug, s: ClipDataset(data, idx, augment=aug, seed=s)  # noqa: E731
        in_dim = 3
    g = torch.Generator().manual_seed(cfg.seed)
    # Small datasets: smaller batches so every epoch still makes enough optimisation steps
    batch_size = int(min(cfg.batch_size, max(8, len(tr) // 8)))
    train_dl = DataLoader(make(tr, True, cfg.seed), batch_size=batch_size, shuffle=True, generator=g)
    val_dl = DataLoader(make(va, False, 0), batch_size=256)
    test_ds = make(te, False, 0)
    test_dl = DataLoader(test_ds, batch_size=256)

    set_seed(cfg.seed)
    model = build_model(spec["arch"], in_dim, len(CLASSES)).to(device)
    # class weights: balanced loss even if some gestures have fewer samples
    counts = np.bincount(data["y"][tr], minlength=len(CLASSES)).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(CLASSES), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)

    log(f"[{name}] {spec['label']} - {count_params(model):,} parameters, "
        f"train/val/test = {len(tr)}/{len(va)}/{len(te)} samples, batch {batch_size}")
    history, best, best_loss, best_state, bad, t0 = [], -1.0, float("inf"), None, 0, time.time()
    best_epoch = 1
    for epoch in range(1, cfg.epochs + 1):
        tl, ta, _, _ = _run_epoch(model, train_dl, loss_fn, device, opt, should_stop)
        vl, vacc, vy, vp = _run_epoch(model, val_dl, loss_fn, device)
        vf1 = metrics.summary(vy, vp, len(CLASSES))["macro_f1"]
        sched.step(vl)   # val loss is smoother than F1 on a small validation set
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "train_acc": ta, "val_acc": vacc,
                        "val_f1": vf1})
        # best checkpoint = highest val macro-F1; on a tie, the lower val loss
        if vf1 > best + 1e-6 or (vf1 >= best - 1e-6 and vl < best_loss):
            best, best_loss, best_state, bad = vf1, vl, copy.deepcopy(model.state_dict()), 0
            best_epoch = epoch  # the epoch whose weights are actually kept
        else:
            bad += 1
        progress(name, epoch, cfg.epochs, history[-1])
        if bad >= cfg.patience and epoch >= cfg.min_epochs:
            log(f"[{name}] early stopping at epoch {epoch} (no val improvement for {cfg.patience} epochs)")
            break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    _, test_acc, ty, tp = _run_epoch(model, test_dl, loss_fn, device)
    s = metrics.summary(ty, tp, len(CLASSES))

    # ---- save checkpoint (+ everything Step 4 needs to rebuild the input pipeline)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"name": name, "arch": spec["arch"], "kind": spec["kind"], "motion": spec["motion"],
                "in_dim": in_dim, "classes": CLASSES, "seq_len": SEQ_LEN, "clip_frames": CLIP_FRAMES,
                "clip_size": CLIP_SIZE, "state_dict": model.cpu().state_dict(), "test_acc": test_acc,
                "test_macro_f1": s["macro_f1"]}, cfg.models_dir / f"gesture_{name}.pt")
    model.to(device)

    # ---- report figures and tables
    out = cfg.out_dir
    plots.plot_curves(history, out / f"{name}_curves.png", spec["label"], best_epoch)
    plots.plot_confusion(s["cm"], CLASSES, out / f"{name}_confusion.png",
                         f"{spec['label']}\ntest acc {test_acc:.3f}, macro-F1 {s['macro_f1']:.3f}")
    prec, rec, f1, sup = metrics.per_class(s["cm"])
    with open(out / f"{name}_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gesture", "precision", "recall", "f1", "support"])
        for c, p_, r_, f_, n_ in zip(CLASSES, prec, rec, f1, sup):
            w.writerow([c, f"{p_:.4f}", f"{r_:.4f}", f"{f_:.4f}", int(n_)])
    with open(out / f"{name}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    result = {"model": name, "description": spec["label"], "params": count_params(model),
              "epochs_run": len(history), "best_epoch": best_epoch,
              "val_macro_f1": round(best, 4), "test_acc": round(test_acc, 4),
              "test_macro_f1": round(s["macro_f1"], 4),
              "latency_ms": round(_latency_ms(model, test_ds[0][0], device), 2),
              "train_time_s": round(train_time, 1)}
    log(f"[{name}] TEST acc {test_acc:.3f} · macro-F1 {s['macro_f1']:.3f} "
        f"(best epoch {best_epoch}, {train_time:.0f}s)")
    return result


def run(cfg=None, log=print, progress=None, should_stop=None):
    """Train every model in cfg.models. Returns the list of result dicts (also written to results.csv)."""
    cfg = cfg or TrainConfig()
    progress = progress or (lambda name, ep, total, h: log(
        f"  [{name}] epoch {ep}/{total}  loss {h['train_loss']:.3f}  val acc {h['val_acc']:.3f}  "
        f"val F1 {h['val_f1']:.3f}"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))  # leave a core for the UI
    set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading dataset from {cfg.data_dir} ...")
    data = load_all(cfg.data_dir, with_crops="cnn" in cfg.models, log=log)
    n = len(data["y"])
    found = np.bincount(data["y"], minlength=len(CLASSES))
    log(f"{n} samples, {len(set(data['persons']))} people - per gesture: "
        + ", ".join(f"{c}={k}" for c, k in zip(CLASSES, found)))
    if n < 30 or (found == 0).any():
        missing = [c for c, k in zip(CLASSES, found) if k == 0]
        raise ValueError(f"Not enough data to train: {n} samples"
                         + (f", no samples for {missing}" if missing else "")
                         + ". Record more in the Data & Training page.")

    split, note = split_by_person(data["persons"], data["y"], cfg.seed)
    log(note)
    (cfg.out_dir / "split.txt").write_text(note + "\n" + "\n".join(
        f"{k}: {len(v)} samples" for k, v in zip(("train", "val", "test"), split)), encoding="utf-8")

    results = []
    for name in cfg.models:
        if MODEL_SPECS[name]["kind"] == "clip" and data["crops"] is None:
            log(f"[{name}] skipped: some samples were recorded without hand crops")
            continue
        try:
            results.append(train_one(name, data, split, cfg, device, log, progress, should_stop))
        except StopTraining:
            log("Training stopped by the user.")
            break

    if results:
        with open(cfg.out_dir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)
        plots.plot_comparison(results, cfg.out_dir / "comparison.png")
        log(f"Results saved to {cfg.out_dir}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", type=Path, default=GESTURE_DIR)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "gestures")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="where to save the .pt checkpoints")
    a = ap.parse_args()
    run(TrainConfig(models=a.models, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed,
                    data_dir=a.data, out_dir=a.out, models_dir=a.models_dir))


if __name__ == "__main__":
    main()
