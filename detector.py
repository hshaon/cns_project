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
    requested = set(feature_subset)
    return [feature for feature in FEATURES if feature in requested]


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


def _threshold_rows_hybrid(
    feature_frame: pd.DataFrame,
    selected_features: List[str],
    threshold_mult: float,
    baseline_window: int,
    min_baseline: int,
    calibration_seconds: int,
) -> pd.DataFrame:
    adaptive_frame = _threshold_rows_adaptive(
        feature_frame,
        selected_features,
        threshold_mult,
        baseline_window,
        min_baseline,
    )
    fixed_frame = _threshold_rows_fixed(
        feature_frame,
        selected_features,
        threshold_mult,
        calibration_seconds,
    )

    hybrid_rows = adaptive_frame.copy()
    for feature in selected_features:
        if feature in SYN_RULE_FEATURES:
            hybrid_rows[feature] = fixed_frame[feature]
        else:
            hybrid_rows[feature] = adaptive_frame[feature]

    return hybrid_rows


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


def _apply_persistence(raw_predictions: List[int], min_alert_windows: int, cooldown_windows: int) -> List[int]:
    if min_alert_windows <= 1 and cooldown_windows <= 0:
        return raw_predictions

    smoothed = [0] * len(raw_predictions)
    active = False
    consecutive_hits = 0
    cooldown_remaining = 0

    for index, hit in enumerate(raw_predictions):
        if hit:
            consecutive_hits += 1
        else:
            consecutive_hits = 0

        if not active and consecutive_hits >= max(1, min_alert_windows):
            active = True
            start_index = max(0, index - min_alert_windows + 1)
            for mark_index in range(start_index, index + 1):
                smoothed[mark_index] = 1

        if active:
            if hit:
                smoothed[index] = 1
                cooldown_remaining = cooldown_windows
            elif cooldown_remaining > 0:
                smoothed[index] = 1
                cooldown_remaining -= 1
            else:
                active = False
                cooldown_remaining = 0

    return smoothed


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
    min_alert_windows: int = 1,
    cooldown_windows: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if features.empty:
        out = features.copy()
        out["raw_prediction"] = pd.Series(dtype=int)
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
    elif detector_mode == "hybrid":
        threshold_frame = _threshold_rows_hybrid(
            features,
            selected_features,
            threshold_mult,
            baseline_window,
            min_baseline,
            calibration_seconds,
        )
    else:
        raise ValueError(f"Unsupported detector mode: {detector_mode}")

    raw_predictions: List[int] = []
    for index in range(len(features)):
        current_row = features.iloc[index]
        threshold_row = threshold_frame.iloc[index]
        if rule_mode == "protocol":
            raw_prediction = _protocol_rule_prediction(current_row, threshold_row, selected_features)
        elif rule_mode == "kofn":
            raw_prediction = _kofn_rule_prediction(current_row, threshold_row, selected_features, kofn_k)
        else:
            raise ValueError(f"Unsupported rule mode: {rule_mode}")
        raw_predictions.append(raw_prediction)

    predictions = _apply_persistence(raw_predictions, min_alert_windows, cooldown_windows)
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
    out["min_alert_windows"] = min_alert_windows
    out["cooldown_windows"] = cooldown_windows
    out["raw_prediction"] = raw_predictions
    out["prediction"] = predictions

    return out, threshold_frame
