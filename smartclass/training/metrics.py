"""Classification metrics written with numpy (no sklearn): confusion matrix, per-class P/R/F1."""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm  # rows = true class, columns = predicted class


def per_class(cm):
    """Precision, recall, F1 and support for each class."""
    tp = np.diag(cm).astype(float)
    pred, true = cm.sum(0), cm.sum(1)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp),
                   where=(precision + recall) > 0)
    return precision, recall, f1, true


def summary(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    _, _, f1, support = per_class(cm)
    present = support > 0  # macro-F1 over the classes that exist in this split
    return {
        "acc": float(np.trace(cm) / max(cm.sum(), 1)),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "cm": cm,
    }
