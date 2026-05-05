from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def evaluate(pred: pd.Series, truth: pd.Series) -> Dict[str, float]:
    pred_values = pred.astype(int).to_numpy()
    truth_values = truth.astype(int).to_numpy()

    tp = int(((pred_values == 1) & (truth_values == 1)).sum())
    tn = int(((pred_values == 0) & (truth_values == 0)).sum())
    fp = int(((pred_values == 1) & (truth_values == 0)).sum())
    fn = int(((pred_values == 0) & (truth_values == 1)).sum())

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


def phase_metrics(detections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, group in detections.groupby("phase"):
        metrics = evaluate(group["prediction"], group["label"])
        phase_kind = "attack" if int(group["label"].max()) == 1 else "benign"
        rows.append(
            {
                "phase": phase,
                "phase_kind": phase_kind,
                "windows": int(len(group)),
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "recall": metrics["recall"],
                "false_positive_rate": metrics["false_positive_rate"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(results: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    metric_columns = ["accuracy", "precision", "recall", "false_positive_rate", "f1"]
    grouped = results.groupby(list(group_cols))[metric_columns].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(part for part in column if part).rstrip("_") if isinstance(column, tuple) else column
        for column in grouped.columns
    ]
    return grouped


def sweep_thresholds(
    features: pd.DataFrame,
    truth: pd.Series,
    detector_fn,
    mult_values,
    detector_mode: str,
    rule_mode: str,
    baseline_window: int,
    min_baseline: int,
    calibration_seconds: int,
    feature_subset,
    kofn_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tprs = []
    fprs = []
    mults = []

    for mult in mult_values:
        detections, _ = detector_fn(
            features,
            detector_mode=detector_mode,
            rule_mode=rule_mode,
            threshold_mult=float(mult),
            baseline_window=baseline_window,
            min_baseline=min_baseline,
            calibration_seconds=calibration_seconds,
            feature_subset=feature_subset,
            kofn_k=kofn_k,
        )
        metrics = evaluate(detections["prediction"], truth)
        tprs.append(metrics["recall"])
        fprs.append(metrics["false_positive_rate"])
        mults.append(mult)

    return np.array(fprs), np.array(tprs), np.array(mults)
