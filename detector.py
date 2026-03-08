from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


FEATURES = [
    "packets_per_sec",
    "bytes_per_sec",
    "syn_per_sec",
    "unique_dst_ports",
    "unique_dst_ips",
]


def detect(
    features: pd.DataFrame,
    baseline_window: int = 30,
    threshold_mult: float = 3.0,
    min_baseline: int = 10,
) -> Tuple[pd.DataFrame, List[Dict[str, float]]]:
    if features.empty:
        out = features.copy()
        out["prediction"] = []
        return out, []

    thresholds: List[Dict[str, float]] = []
    predictions: List[int] = []

    values = features[FEATURES].astype(float).to_numpy()

    for i in range(len(features)):
        start = max(0, i - baseline_window)
        end = i
        if end - start < min_baseline:
            start = 0
            end = max(min_baseline, i)
        baseline = values[start:end]
        if baseline.size == 0:
            baseline = values[:min_baseline]

        mean = baseline.mean(axis=0)
        std = baseline.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        thresh = mean + threshold_mult * std

        current = values[i]
        is_attack = int(np.any(current > thresh))

        thresholds.append({FEATURES[j]: float(thresh[j]) for j in range(len(FEATURES))})
        predictions.append(is_attack)

    out = features.copy()
    out["prediction"] = predictions
    return out, thresholds
