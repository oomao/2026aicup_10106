"""Post-processing: posthoc logit adjustment + coordinate-descent thresholds.
Same core logic as baseline, but tau grid is denser for finer adjustment."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def posthoc_logit_adjust(probs: np.ndarray, prior: np.ndarray, tau: float):
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    logits = logits - tau * np.log(np.clip(prior, 1e-12, 1.0))
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def macro_f1(y_true, y_pred, n_classes):
    return f1_score(y_true, y_pred,
                    labels=list(range(n_classes)),
                    average='macro', zero_division=0)


def tune_tau(probs, y, prior, n_classes,
             taus=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0)):
    best_tau, best_f1 = 0.0, -1.0
    for t in taus:
        adj = posthoc_logit_adjust(probs, prior, t)
        f1 = macro_f1(y, adj.argmax(axis=1), n_classes)
        if f1 > best_f1:
            best_f1, best_tau = f1, t
    return best_tau, best_f1


def coordinate_descent_thresholds(probs, y, n_classes, n_iter=4, grid=None):
    if grid is None:
        grid = np.linspace(-5, 5, 41)
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    shifts = np.zeros(n_classes)

    def f1_with(sh):
        return macro_f1(y, (logits + sh).argmax(axis=1), n_classes)

    best_f1 = f1_with(shifts)
    for _ in range(n_iter):
        improved = False
        for k in range(n_classes):
            orig = shifts[k]
            best_local, best_shift = best_f1, orig
            for cand in grid:
                shifts[k] = cand
                f1 = f1_with(shifts)
                if f1 > best_local:
                    best_local, best_shift = f1, cand
            shifts[k] = best_shift
            if best_local > best_f1 + 1e-6:
                best_f1 = best_local
                improved = True
        if not improved:
            break
    return shifts, best_f1


def apply_shifts(probs, shifts):
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    return (logits + shifts).argmax(axis=1)
