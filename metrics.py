"""Metrics named in the rubric."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def precision_at_k(y, p, k):
    order = np.argsort(-np.asarray(p), kind="stable")[:k]
    return float(np.asarray(y)[order].mean())


def recall_at_k(y, p, k):
    order = np.argsort(-np.asarray(p), kind="stable")[:k]
    return float(np.asarray(y)[order].sum() / max(np.asarray(y).sum(), 1))


def ece(y, p, bins=5):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def classification_report(y, p, k):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            f"precision@{k}": precision_at_k(y, p, k), f"recall@{k}": recall_at_k(y, p, k),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)),
            "ece": ece(y, p)}
