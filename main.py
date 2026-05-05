import argparse
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from detector import FEATURES, detect
from evaluator import aggregate_metrics, evaluate, phase_metrics, sweep_thresholds
from feature_extractor import FEATURE_COLUMNS, extract_features
from mitigation import simulate_mitigation
from traffic_generator import generate_pcap
from visualizer import (
    plot_ablation_comparison,
    plot_detector_comparison,
    plot_multi_seed_summary,
    plot_phase_metrics,
    plot_roc,
    plot_timeseries,
)


DEFAULT_PCAP = "pcaps/iot_traffic.pcap"
DEFAULT_LABELS = "results/labels.csv"
DEFAULT_FEATURE_SETS: Dict[str, List[str]] = {
    "packets_only": ["packets_per_sec"],
    "packets_bytes": ["packets_per_sec", "bytes_per_sec"],
    "syn_ports": ["syn_per_sec", "unique_dst_ports"],
    "full": FEATURES,
}


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _seed_values(base_seed: int, total_seeds: int) -> List[int]:
    return [base_seed + offset for offset in range(total_seeds)]


def _parse_csv_list(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _write_metrics_txt(metrics: Dict[str, float], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for key, value in metrics.items():
            handle.write(f"{key}: {value}\n")


def _build_merged_features(labels_path: str, pcap_path: str, window_s: int) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    features = extract_features(pcap_path, window_s=window_s, total_windows=len(labels))
    merged = labels.merge(features, on="window_start", how="left")
    fill_map = {column: 0.0 for column in FEATURE_COLUMNS}
    merged = merged.fillna(fill_map)
    return merged


def _dataset_for_seed(args: argparse.Namespace, seed: int, pcap_path: str, labels_path: str) -> pd.DataFrame:
    generate_pcap(pcap_path, labels_path, seed=seed)
    return _build_merged_features(labels_path, pcap_path, args.window)


def run_single_pipeline(
    args: argparse.Namespace,
    seed: int,
    detector_mode: str,
    rule_mode: str,
    feature_set_name: str,
    feature_subset: Sequence[str],
    pcap_path: str,
    labels_path: str,
    write_outputs: bool,
) -> Dict[str, object]:
    if args.generate or not os.path.exists(pcap_path) or not os.path.exists(labels_path):
        generate_pcap(pcap_path, labels_path, seed=seed)

    merged = _build_merged_features(labels_path, pcap_path, args.window)
    detections, threshold_frame = detect(
        merged,
        detector_mode=detector_mode,
        rule_mode=rule_mode,
        threshold_mult=args.threshold_mult,
        baseline_window=args.baseline_window,
        min_baseline=args.min_baseline,
        calibration_seconds=args.calibration_seconds,
        feature_subset=feature_subset,
        kofn_k=args.kofn_k,
    )
    metrics = evaluate(detections["prediction"], detections["label"])
    phase_frame = phase_metrics(detections)

    if write_outputs:
        detections.to_csv(os.path.join(args.results_dir, "detections.csv"), index=False)
        merged.to_csv(os.path.join(args.results_dir, "features.csv"), index=False)
        threshold_frame.to_csv(os.path.join(args.results_dir, "thresholds.csv"), index=False)
        phase_frame.to_csv(os.path.join(args.results_dir, "phase_metrics_single_run.csv"), index=False)
        _write_metrics_txt(metrics, os.path.join(args.results_dir, "metrics.txt"))
        simulate_mitigation(detections, os.path.join(args.results_dir, "mitigation.log"))
        plot_timeseries(merged, merged["label"], os.path.join(args.results_dir, "timeseries.png"))

        if args.sweep:
            mults = np.arange(args.sweep_min, args.sweep_max + 1e-9, args.sweep_step)
            fprs, tprs, mult_vals = sweep_thresholds(
                merged,
                merged["label"],
                detect,
                mults,
                detector_mode=detector_mode,
                rule_mode=rule_mode,
                baseline_window=args.baseline_window,
                min_baseline=args.min_baseline,
                calibration_seconds=args.calibration_seconds,
                feature_subset=feature_subset,
                kofn_k=args.kofn_k,
            )
            plot_roc(fprs, tprs, os.path.join(args.results_dir, "roc_like.png"), list(mult_vals))

    return {
        "merged": merged,
        "detections": detections,
        "metrics": metrics,
        "phase_metrics": phase_frame,
        "feature_set_name": feature_set_name,
        "detector_mode": detector_mode,
        "rule_mode": rule_mode,
        "seed": seed,
    }


def _results_record(
    run_result: Dict[str, object],
    threshold_mult: float,
    baseline_window: int,
) -> Dict[str, object]:
    metrics = dict(run_result["metrics"])
    return {
        "seed": run_result["seed"],
        "detector_mode": run_result["detector_mode"],
        "rule_mode": run_result["rule_mode"],
        "feature_set": run_result["feature_set_name"],
        "threshold_mult": threshold_mult,
        "baseline_window": baseline_window,
        **metrics,
    }


def run_experiments(args: argparse.Namespace) -> None:
    seeds = _seed_values(args.seed, args.seeds)
    detector_modes = _parse_csv_list(args.experiment_detector_modes)
    rule_modes = _parse_csv_list(args.experiment_rule_modes)

    per_run_records: List[Dict[str, object]] = []
    phase_records: List[pd.DataFrame] = []
    ablation_records: List[Dict[str, object]] = []

    ablation_detector_mode = args.ablation_detector_mode
    ablation_rule_mode = args.ablation_rule_mode

    for seed in seeds:
        dataset = _dataset_for_seed(
            args,
            seed,
            os.path.join(args.results_dir, f"temp_seed_{seed}.pcap"),
            os.path.join(args.results_dir, f"temp_seed_{seed}.labels.csv"),
        )

        for detector_mode in detector_modes:
            for rule_mode in rule_modes:
                detections, _ = detect(
                    dataset,
                    detector_mode=detector_mode,
                    rule_mode=rule_mode,
                    threshold_mult=args.threshold_mult,
                    baseline_window=args.baseline_window,
                    min_baseline=args.min_baseline,
                    calibration_seconds=args.calibration_seconds,
                    feature_subset=DEFAULT_FEATURE_SETS["full"],
                    kofn_k=args.kofn_k,
                )
                run_result = {
                    "seed": seed,
                    "detector_mode": detector_mode,
                    "rule_mode": rule_mode,
                    "feature_set_name": "full",
                    "metrics": evaluate(detections["prediction"], detections["label"]),
                    "phase_metrics": phase_metrics(detections),
                }
                per_run_records.append(_results_record(run_result, args.threshold_mult, args.baseline_window))
                phase_frame = run_result["phase_metrics"].copy()
                phase_frame["seed"] = seed
                phase_frame["detector_mode"] = detector_mode
                phase_frame["rule_mode"] = rule_mode
                phase_frame["feature_set"] = "full"
                phase_records.append(phase_frame)

        for feature_set_name, feature_subset in DEFAULT_FEATURE_SETS.items():
            detections, _ = detect(
                dataset,
                detector_mode=ablation_detector_mode,
                rule_mode=ablation_rule_mode,
                threshold_mult=args.threshold_mult,
                baseline_window=args.baseline_window,
                min_baseline=args.min_baseline,
                calibration_seconds=args.calibration_seconds,
                feature_subset=feature_subset,
                kofn_k=args.kofn_k,
            )
            run_result = {
                "seed": seed,
                "detector_mode": ablation_detector_mode,
                "rule_mode": ablation_rule_mode,
                "feature_set_name": feature_set_name,
                "metrics": evaluate(detections["prediction"], detections["label"]),
            }
            ablation_records.append(_results_record(run_result, args.threshold_mult, args.baseline_window))

    per_run_metrics = pd.DataFrame(per_run_records)
    summary_metrics = aggregate_metrics(
        per_run_metrics,
        ["detector_mode", "rule_mode", "feature_set", "threshold_mult", "baseline_window"],
    )

    raw_phase_metrics = pd.concat(phase_records, ignore_index=True)
    phase_summary = (
        raw_phase_metrics.groupby(["detector_mode", "rule_mode", "feature_set", "phase", "phase_kind"])[
            ["recall", "false_positive_rate", "accuracy", "precision", "f1"]
        ]
        .mean()
        .reset_index()
    )

    ablation_metrics = pd.DataFrame(ablation_records)

    per_run_metrics.to_csv(os.path.join(args.results_dir, "per_run_metrics.csv"), index=False)
    summary_metrics.to_csv(os.path.join(args.results_dir, "summary_metrics.csv"), index=False)
    phase_summary.to_csv(os.path.join(args.results_dir, "phase_metrics.csv"), index=False)
    ablation_metrics.to_csv(os.path.join(args.results_dir, "ablation_metrics.csv"), index=False)

    plot_detector_comparison(summary_metrics, os.path.join(args.results_dir, "detector_comparison.png"))
    plot_phase_metrics(phase_summary, os.path.join(args.results_dir, "phase_metrics.png"))
    plot_multi_seed_summary(summary_metrics, os.path.join(args.results_dir, "multi_seed_summary.png"))
    plot_ablation_comparison(ablation_metrics, os.path.join(args.results_dir, "ablation_comparison.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IoT DoS detection pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic traffic PCAP")
    parser.add_argument("--pcap", default=DEFAULT_PCAP, help="Path to PCAP file")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Path to labels CSV")
    parser.add_argument("--results-dir", default="results", help="Output directory")
    parser.add_argument("--window", type=int, default=1, help="Window size in seconds")
    parser.add_argument("--baseline-window", type=int, default=30, help="Rolling baseline window in seconds")
    parser.add_argument("--min-baseline", type=int, default=10, help="Minimum baseline windows")
    parser.add_argument("--threshold-mult", type=float, default=3.0, help="Threshold multiplier")
    parser.add_argument("--detector-mode", default="adaptive", choices=["adaptive", "fixed"], help="Detector mode")
    parser.add_argument("--rule-mode", default="protocol", choices=["protocol", "kofn"], help="Decision rule mode")
    parser.add_argument("--calibration-seconds", type=int, default=60, help="Calibration period for fixed thresholds")
    parser.add_argument("--kofn-k", type=int, default=2, help="Number of threshold hits required for kofn mode")
    parser.add_argument("--seed", type=int, default=1337, help="Base random seed")
    parser.add_argument("--run-experiments", action="store_true", help="Run multi-seed comparison experiments")
    parser.add_argument("--seeds", type=int, default=10, help="Number of sequential seeds to evaluate")
    parser.add_argument("--experiment-detector-modes", default="adaptive,fixed", help="Detector modes for experiments")
    parser.add_argument("--experiment-rule-modes", default="protocol", help="Rule modes for experiments")
    parser.add_argument("--ablation-detector-mode", default="adaptive", choices=["adaptive", "fixed"], help="Detector mode for ablation experiments")
    parser.add_argument("--ablation-rule-mode", default="protocol", choices=["protocol", "kofn"], help="Rule mode for ablation experiments")
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep for the primary single run")
    parser.add_argument("--sweep-min", type=float, default=1.0, help="Sweep min multiplier")
    parser.add_argument("--sweep-max", type=float, default=5.0, help="Sweep max multiplier")
    parser.add_argument("--sweep-step", type=float, default=0.5, help="Sweep step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dir(args.results_dir)
    _ensure_dir(os.path.dirname(args.pcap))
    _ensure_dir(os.path.dirname(args.labels))

    run_single_pipeline(
        args=args,
        seed=args.seed,
        detector_mode=args.detector_mode,
        rule_mode=args.rule_mode,
        feature_set_name="full",
        feature_subset=DEFAULT_FEATURE_SETS["full"],
        pcap_path=args.pcap,
        labels_path=args.labels,
        write_outputs=True,
    )

    if args.run_experiments:
        run_experiments(args)


if __name__ == "__main__":
    main()
