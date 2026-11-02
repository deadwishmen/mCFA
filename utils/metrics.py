"""Evaluation metrics for EMOTIC: Average Precision (mAP), VAD MAE, and threshold calibration."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def test_scikit_ap(cat_preds: np.ndarray, cat_labels: np.ndarray) -> float:
    """Calculate mean Average Precision (mAP) across the 26 discrete emotion categories."""
    ap = np.zeros(26, dtype=np.float32)
    for i in range(26):
        ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
    print("ap", ap, ap.shape, ap.mean())
    return float(ap.mean())


def test_emotic_vad(cont_preds: np.ndarray, cont_labels: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE) across the 3 continuous VAD dimensions."""
    vad = np.zeros(3, dtype=np.float32)
    for i in range(3):
        vad[i] = np.mean(np.abs(cont_preds[i, :] - cont_labels[i, :]))
    print("vad", vad, vad.shape, vad.mean())
    return float(vad.mean())


def get_thresholds(cat_preds: np.ndarray, cat_labels: np.ndarray, save_path: str = "./thresholds.npy") -> np.ndarray:
    """Compute per-class classification thresholds where Precision equals Recall."""
    thresholds = np.zeros(26, dtype=np.float32)
    for i in range(26):
        p, r, t = precision_recall_curve(cat_labels[i, :], cat_preds[i, :])
        for k in range(len(p)):
            if p[k] == r[k] and k < len(t):
                thresholds[i] = t[k]
                break
    if save_path:
        np.save(save_path, thresholds)
    return thresholds
