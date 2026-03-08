from typing import Dict, Tuple

import numpy as np
import pandas as pd


def evaluate(pred: pd.Series, truth: pd.Series) -> Dict[str, float]:
    pred = pred.astype(int).to_numpy()
    truth = truth.astype(int).to_numpy()

    tp = int(((pred == 1) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "f1": f1,
    }


def sweep_thresholds(
    features: pd.DataFrame,
    truth: pd.Series,
    detector_fn,
    mult_values,
    baseline_window: int,
    min_baseline: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tprs = []
    fprs = []
    mults = []

    for mult in mult_values:
        det, _ = detector_fn(
            features,
            baseline_window=baseline_window,
            threshold_mult=float(mult),
            min_baseline=min_baseline,
        )
        metrics = evaluate(det["prediction"], truth)
        tprs.append(metrics["recall"])
        fprs.append(metrics["false_positive_rate"])
        mults.append(mult)

    return np.array(fprs), np.array(tprs), np.array(mults)
