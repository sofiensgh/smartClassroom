"""Step 7 - train the stress models on the pilot dataset and compare single modality vs fusion.

    python -m smartclass.training.train_stress

Methodology
    - inputs: 20 s windows every 2 s (stress/features.py, same code as live)
    - CROSS-VALIDATION BY SESSION: fold k tests on the k-th rest session + the k-th stress session, trains on
      all the others. Windows of one session overlap, so they never go to both train and test (no leakage).
    - 3 models with the same training: face only (LSTM), pulse only (1D CNN), fusion (both) -> ablation
    - fixed seeds, class balance by construction, metrics: accuracy, F1 (stress), ROC AUC
    - finally each model is retrained on ALL sessions -> models/stress_<mode>.pt for the live page
Outputs (output/stress/): results.csv, folds.csv, comparison.png, confusion_fusion.png, timeline.png
"""
import argparse
import csv
import random

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..config import MODELS_DIR, STRESS_DIR, STRESS_OUT, STRESS_STEP_S
from ..models.stress_models import StressNet
from ..stress.dataset import list_sessions, load_session
from ..stress.features import session_windows

MODES = ("face", "pulse", "fusion")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_windows(root, log):
    sessions = []
    for path, person, label in list_sessions(root):
        d = load_session(path)
        f, p, ends = session_windows(d, STRESS_STEP_S)
        if len(f) == 0:
            log(f"  {path.name}: no usable window, skipped")
            continue
        sessions.append({"name": f"{person}/{path.stem}", "label": label, "y": float(label == "stress"),
                         "face": f, "pulse": p, "ends": np.array(ends)})
        log(f"  {person}/{path.stem}: {len(f)} windows")
    return sessions


def stack(sessions):
    return (np.concatenate([s["face"] for s in sessions]), np.concatenate([s["pulse"] for s in sessions]),
            np.concatenate([np.full(len(s["face"]), s["y"], np.float32) for s in sessions]))


def fit(mode, train, epochs, seed, batch=32):
    set_seed(seed)
    f, p, y = stack(train)
    mean, std = f.mean((0, 1)), np.maximum(f.std((0, 1)), 1e-2)
    fx = torch.from_numpy(((f - mean) / std).astype(np.float32))
    px, yx = torch.from_numpy(p), torch.from_numpy(y)
    model = StressNet(mode)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    history = []
    for _ in range(epochs):
        model.train()
        perm, losses = torch.randperm(len(yx)), []
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            fb = fx[idx] + 0.1 * torch.randn_like(fx[idx])                    # augmentation: noise
            pb = torch.roll(px[idx], int(np.random.randint(-15, 16)), 1)       # small time shift
            loss = loss_fn(model(fb, pb), yx[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
    model.eval()
    return model, mean, std, history


@torch.no_grad()
def predict(model, mean, std, sessions):
    f, p, _ = stack(sessions)
    return torch.sigmoid(model(torch.from_numpy(((f - mean) / std).astype(np.float32)), torch.from_numpy(p))).numpy()


def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def scores(y, prob):
    pred = (prob >= 0.5).astype(float)
    tp, fp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"accuracy": float((pred == y).mean()), "f1": float(f1), "auc": auc(y, prob)}


def _fig(path, draw, size):
    fig = Figure(figsize=size)
    FigureCanvasAgg(fig)
    draw(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def run(root=STRESS_DIR, out=STRESS_OUT, models_dir=MODELS_DIR, epochs=30, seed=42, log=print):
    out.mkdir(parents=True, exist_ok=True)
    log("Loading sessions (20 s windows every 2 s)...")
    sessions = load_windows(root, log)
    rest = [s for s in sessions if s["label"] == "rest"]
    stress = [s for s in sessions if s["label"] == "stress"]
    k = min(len(rest), len(stress))
    if k < 2:
        raise ValueError(f"need at least 2 rest + 2 stress sessions (have {len(rest)} rest, {len(stress)} stress)")
    persons = sorted({s["name"].split("/")[0] for s in sessions})
    log(f"{len(sessions)} sessions ({len(rest)} rest, {len(stress)} stress), {len(persons)} person(s): {persons}")
    log(f"Cross-validation: {k} folds, each tests 1 rest + 1 stress session never seen in training")

    fold_rows, all_y, all_p, timeline = [], {m: [] for m in MODES}, {m: [] for m in MODES}, []
    for i in range(k):
        test = [rest[i], stress[i]]
        train = [s for s in sessions if s not in test]
        y = stack(test)[2]
        for mode in MODES:
            model, mean, std, _ = fit(mode, train, epochs, seed + i)
            prob = predict(model, mean, std, test)
            sc = scores(y, prob)
            fold_rows.append({"fold": i + 1, "mode": mode, "test": f"{test[0]['name']} + {test[1]['name']}", **sc})
            all_y[mode].append(y)
            all_p[mode].append(prob)
            if mode == "fusion" and i == 0:
                n0 = len(test[0]["face"])
                timeline = [(test[0], prob[:n0]), (test[1], prob[n0:])]
            log(f"fold {i + 1}/{k}  {mode:6s}  acc {sc['accuracy']:.2f}  F1 {sc['f1']:.2f}  AUC {sc['auc']:.2f}")

    results = []
    for mode in MODES:
        rows = [r for r in fold_rows if r["mode"] == mode]
        res = {"model": {"face": "face only (LSTM)", "pulse": "pulse only (1D CNN)",
                         "fusion": "fusion (LSTM + 1D CNN)"}[mode]}
        for m in ("accuracy", "f1", "auc"):
            v = np.array([r[m] for r in rows], float)
            res[m] = float(np.nanmean(v))
            res[m + "_std"] = float(np.nanstd(v))
        res["folds"], res["sessions"], res["persons"] = k, len(sessions), len(persons)
        results.append(res)
        log(f"CV {res['model']:24s} acc {res['accuracy']:.2f}±{res['accuracy_std']:.2f}  "
            f"F1 {res['f1']:.2f}  AUC {res['auc']:.2f}")

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows([{k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in r.items()} for r in results])
    with open(out / "folds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0]))
        w.writeheader()
        w.writerows([{k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in r.items()} for r in fold_rows])

    def comparison(fig):
        ax = fig.subplots()
        x = np.arange(len(results))
        for j, (m, c) in enumerate((("accuracy", "tab:blue"), ("f1", "tab:orange"), ("auc", "tab:green"))):
            ax.bar(x + (j - 1) * 0.25, [r[m] for r in results], 0.25, yerr=[r[m + "_std"] for r in results],
                   color=c, label=m, capsize=3)
        ax.set_xticks(x, [r["model"] for r in results])
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set(ylim=(0, 1.05), title=f"Stress: single modality vs fusion ({k}-fold CV by session)")
        ax.legend()
    _fig(out / "comparison.png", comparison, (8, 4))

    yf, pf = np.concatenate(all_y["fusion"]), np.concatenate(all_p["fusion"])
    cm = np.array([[((yf == a) & ((pf >= 0.5) == b)).sum() for b in (0, 1)] for a in (0, 1)])

    def confusion(fig):
        ax = fig.subplots()
        ax.imshow(cm, cmap="Blues")
        for a in range(2):
            for b in range(2):
                ax.text(b, a, int(cm[a, b]), ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], ["rest", "stress"])
        ax.set_yticks([0, 1], ["rest", "stress"])
        ax.set(xlabel="predicted", ylabel="true", title="Fusion model, all CV test windows")
    _fig(out / "confusion_fusion.png", confusion, (4.5, 4))

    def tl(fig):
        ax = fig.subplots()
        for s, p in timeline:
            ax.plot(s["ends"], p, label=f"{s['name']} ({s['label']})", color="tab:red" if s["y"] else "tab:blue")
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set(xlabel="time in session (s)", ylabel="P(stress)", ylim=(0, 1), title="Fusion model on unseen sessions (fold 1)")
        ax.legend()
    _fig(out / "timeline.png", tl, (8, 3.5))

    log("Training the final models on ALL sessions (for the live page)...")
    models_dir.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        model, mean, std, _ = fit(mode, sessions, epochs, seed)
        torch.save({"state_dict": model.state_dict(), "mode": mode, "mean": torch.from_numpy(mean),
                    "std": torch.from_numpy(std), "cv": next(r for r in results if r["model"].startswith(
                        {"face": "face", "pulse": "pulse", "fusion": "fusion"}[mode]))},
                   models_dir / f"stress_{mode}.pt")
    log(f"Saved models/stress_*.pt and figures in {out}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(epochs=a.epochs, seed=a.seed)


if __name__ == "__main__":
    main()
