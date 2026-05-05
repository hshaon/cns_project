from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


FEATURES = [
    "packets_per_sec",
    "bytes_per_sec",
    "syn_per_sec",
    "unique_dst_ports",
    "unique_dst_ips",
]

UDP_RULE_FEATURES = ["packets_per_sec", "bytes_per_sec"]
SYN_RULE_FEATURES = ["syn_per_sec", "unique_dst_ports"]


def _resolve_feature_subset(feature_subset: Optional[Sequence[str]]) -> List[str]:
    if feature_subset is None:
        return FEATURES.copy()
    return [feature for feature in FEATURES if feature in set(feature_subset)]


def _threshold_rows_adaptive(
    feature_frame: pd.DataFrame,
    selected_features: List[str],
    threshold_mult: float,
    baseline_window: int,
    min_baseline: int,
) -> pd.DataFrame:
    values = feature_frame[selected_features].astype(float).to_numpy()
    rows: List[Dict[str, float]] = []

    for index in range(len(feature_frame)):
        start = max(0, index - baseline_window)
        end = index
        if end - start < min_baseline:
            start = 0
            end = max(min_baseline, index)

        baseline = values[start:end]
        if baseline.size == 0:
            baseline = values[:max(min_baseline, 1)]

        mean = baseline.mean(axis=0)
        std = baseline.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        threshold = mean + threshold_mult * std
        rows.append({selected_features[idx]: float(threshold[idx]) for idx in range(len(selected_features))})

    return pd.DataFrame(rows)


def _threshold_rows_fixed(
    feature_frame: pd.DataFrame,
    selected_features: List[str],
    threshold_mult: float,
    calibration_seconds: int,
) -> pd.DataFrame:
    calibration = feature_frame.iloc[:calibration_seconds][selected_features].astype(float)
    if calibration.empty:
        calibration = feature_frame.iloc[:1][selected_features].astype(float)

    mean = calibration.mean(axis=0)
    std = calibration.std(axis=0).fillna(1.0)
    std = std.where(std >= 1e-6, 1.0)
    threshold = mean + threshold_mult * std
    threshold_row = threshold.to_dict()
    return pd.DataFrame([threshold_row for _ in range(len(feature_frame))])


def _protocol_rule_prediction(
    current_row: pd.Series,
    threshold_row: pd.Series,
    selected_features: List[str],
) -> int:
    selected = set(selected_features)

    udp_features = [feature for feature in UDP_RULE_FEATURES if feature in selected]
    syn_features = [feature for feature in SYN_RULE_FEATURES if feature in selected]

    udp_trigger = bool(udp_features) and all(current_row[feature] > threshold_row[feature] for feature in udp_features)
    syn_trigger = bool(syn_features) and all(current_row[feature] > threshold_row[feature] for feature in syn_features)

    return int(udp_trigger or syn_trigger)


def _kofn_rule_prediction(
    current_row: pd.Series,
    threshold_row: pd.Series,
    selected_features: List[str],
    k: int,
) -> int:
    required = max(1, min(k, len(selected_features)))
    hits = sum(current_row[feature] > threshold_row[feature] for feature in selected_features)
    return int(hits >= required)


def detect(
    features: pd.DataFrame,
    detector_mode: str = "adaptive",
    rule_mode: str = "protocol",
    threshold_mult: float = 3.0,
    baseline_window: int = 30,
    min_baseline: int = 10,
    calibration_seconds: int = 60,
    feature_subset: Optional[Sequence[str]] = None,
    kofn_k: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if features.empty:
        out = features.copy()
        out["prediction"] = pd.Series(dtype=int)
        return out, pd.DataFrame()

    selected_features = _resolve_feature_subset(feature_subset)
    if not selected_features:
        raise ValueError("feature_subset must contain at least one supported feature")

    if detector_mode == "adaptive":
        threshold_frame = _threshold_rows_adaptive(
            features,
            selected_features,
            threshold_mult,
            baseline_window,
            min_baseline,
        )
    elif detector_mode == "fixed":
        threshold_frame = _threshold_rows_fixed(
            features,
            selected_features,
            threshold_mult,
            calibration_seconds,
        )
    else:
        raise ValueError(f"Unsupported detector mode: {detector_mode}")

    predictions: List[int] = []
    for index in range(len(features)):
        current_row = features.iloc[index]
        threshold_row = threshold_frame.iloc[index]
        if rule_mode == "protocol":
            prediction = _protocol_rule_prediction(current_row, threshold_row, selected_features)
        elif rule_mode == "kofn":
            prediction = _kofn_rule_prediction(current_row, threshold_row, selected_features, kofn_k)
        else:
            raise ValueError(f"Unsupported rule mode: {rule_mode}")
        predictions.append(prediction)

    threshold_columns = {
        f"{feature}_threshold": threshold_frame[feature].to_numpy()
        for feature in selected_features
    }

    out = features.copy()
    for feature in FEATURES:
        out[f"{feature}_selected"] = int(feature in selected_features)
    for column_name, values in threshold_columns.items():
        out[column_name] = values
    out["detector_mode"] = detector_mode
    out["rule_mode"] = rule_mode
    out["prediction"] = predictions

    return out, threshold_frame
