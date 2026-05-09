import argparse
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from detector import FEATURES, detect
from evaluator import aggregate_metrics, metrics_with_phase_context, phase_metrics, sweep_thresholds
from feature_extractor import FEATURE_COLUMNS, extract_features
from mitigation import simulate_mitigation
from traffic_generator import generate_pcap
from visualizer import (
    plot_ablation_comparison,
    plot_detector_comparison,
    plot_final_phase_breakdown,
    plot_final_tradeoff_bars,
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


def _config_name(detector_mode: str, rule_mode: str, min_alert_windows: int, cooldown_windows: int) -> str:
    if detector_mode == "hybrid" and min_alert_windows > 1:
        return "hybrid_persistent"
    return detector_mode


def _detect_dataset(
    dataset: pd.DataFrame,
    detector_mode: str,
    rule_mode: str,
    feature_subset: Sequence[str],
    args: argparse.Namespace,
    min_alert_windows: int,
    cooldown_windows: int,
):
    return detect(
        dataset,
        detector_mode=detector_mode,
        rule_mode=rule_mode,
        threshold_mult=args.threshold_mult,
        baseline_window=args.baseline_window,
        min_baseline=args.min_baseline,
        calibration_seconds=args.calibration_seconds,
        feature_subset=feature_subset,
        kofn_k=args.kofn_k,
        min_alert_windows=min_alert_windows,
        cooldown_windows=cooldown_windows,
    )


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
    min_alert_windows: int,
    cooldown_windows: int,
) -> Dict[str, object]:
    if args.generate or not os.path.exists(pcap_path) or not os.path.exists(labels_path):
        generate_pcap(pcap_path, labels_path, seed=seed)

    merged = _build_merged_features(labels_path, pcap_path, args.window)
    detections, threshold_frame = _detect_dataset(
        merged,
        detector_mode,
        rule_mode,
        feature_subset,
        args,
        min_alert_windows,
        cooldown_windows,
    )
    metrics = metrics_with_phase_context(detections)
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
                min_alert_windows=min_alert_windows,
                cooldown_windows=cooldown_windows,
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
        "config_name": _config_name(detector_mode, rule_mode, min_alert_windows, cooldown_windows),
        "min_alert_windows": min_alert_windows,
        "cooldown_windows": cooldown_windows,
    }


def _results_record(
    run_result: Dict[str, object],
    threshold_mult: float,
    baseline_window: int,
) -> Dict[str, object]:
    metrics = dict(run_result["metrics"])
    return {
        "seed": run_result["seed"],
        "config_name": run_result["config_name"],
        "detector_mode": run_result["detector_mode"],
        "rule_mode": run_result["rule_mode"],
        "feature_set": run_result["feature_set_name"],
        "min_alert_windows": run_result["min_alert_windows"],
        "cooldown_windows": run_result["cooldown_windows"],
        "threshold_mult": threshold_mult,
        "baseline_window": baseline_window,
        **metrics,
    }


def _experiment_configs(args: argparse.Namespace) -> List[Dict[str, object]]:
    return [
        {
            "config_name": "adaptive",
            "detector_mode": "adaptive",
            "rule_mode": "protocol",
            "min_alert_windows": 1,
            "cooldown_windows": 0,
        },
        {
            "config_name": "fixed",
            "detector_mode": "fixed",
            "rule_mode": "protocol",
            "min_alert_windows": 1,
            "cooldown_windows": 0,
        },
        {
            "config_name": "hybrid",
            "detector_mode": "hybrid",
            "rule_mode": "protocol",
            "min_alert_windows": 1,
            "cooldown_windows": 0,
        },
        {
            "config_name": "hybrid_persistent",
            "detector_mode": "hybrid",
            "rule_mode": "protocol",
            "min_alert_windows": max(2, args.min_alert_windows),
            "cooldown_windows": 0,
        },
    ]


def run_experiments(args: argparse.Namespace) -> None:
    seeds = _seed_values(args.seed, args.seeds)
    experiment_configs = _experiment_configs(args)

    per_run_records: List[Dict[str, object]] = []
    phase_records: List[pd.DataFrame] = []
    ablation_records: List[Dict[str, object]] = []

    for seed in seeds:
        dataset = _dataset_for_seed(
            args,
            seed,
            os.path.join(args.results_dir, f"temp_seed_{seed}.pcap"),
            os.path.join(args.results_dir, f"temp_seed_{seed}.labels.csv"),
        )

        for config in experiment_configs:
            detections, _ = _detect_dataset(
                dataset,
                config["detector_mode"],
                config["rule_mode"],
                DEFAULT_FEATURE_SETS["full"],
                args,
                int(config["min_alert_windows"]),
                int(config["cooldown_windows"]),
            )
            run_result = {
                "seed": seed,
                "detector_mode": config["detector_mode"],
                "rule_mode": config["rule_mode"],
                "feature_set_name": "full",
                "metrics": metrics_with_phase_context(detections),
                "phase_metrics": phase_metrics(detections),
                "config_name": config["config_name"],
                "min_alert_windows": int(config["min_alert_windows"]),
                "cooldown_windows": int(config["cooldown_windows"]),
            }
            per_run_records.append(_results_record(run_result, args.threshold_mult, args.baseline_window))
            phase_frame = run_result["phase_metrics"].copy()
            phase_frame["seed"] = seed
            phase_frame["config_name"] = config["config_name"]
            phase_frame["detector_mode"] = config["detector_mode"]
            phase_frame["rule_mode"] = config["rule_mode"]
            phase_frame["feature_set"] = "full"
            phase_frame["min_alert_windows"] = int(config["min_alert_windows"])
            phase_frame["cooldown_windows"] = int(config["cooldown_windows"])
            phase_records.append(phase_frame)

        for feature_set_name, feature_subset in DEFAULT_FEATURE_SETS.items():
            detections, _ = _detect_dataset(
                dataset,
                args.ablation_detector_mode,
                args.ablation_rule_mode,
                feature_subset,
                args,
                args.min_alert_windows,
                args.cooldown_windows,
            )
            run_result = {
                "seed": seed,
                "detector_mode": args.ablation_detector_mode,
                "rule_mode": args.ablation_rule_mode,
                "feature_set_name": feature_set_name,
                "metrics": metrics_with_phase_context(detections),
                "config_name": _config_name(
                    args.ablation_detector_mode,
                    args.ablation_rule_mode,
                    args.min_alert_windows,
                    args.cooldown_windows,
                ),
                "min_alert_windows": args.min_alert_windows,
                "cooldown_windows": args.cooldown_windows,
            }
            ablation_records.append(_results_record(run_result, args.threshold_mult, args.baseline_window))

    per_run_metrics = pd.DataFrame(per_run_records)
    summary_metrics = aggregate_metrics(
        per_run_metrics,
        [
            "config_name",
            "detector_mode",
            "rule_mode",
            "feature_set",
            "min_alert_windows",
            "cooldown_windows",
            "threshold_mult",
            "baseline_window",
        ],
    )

    raw_phase_metrics = pd.concat(phase_records, ignore_index=True)
    phase_summary = (
        raw_phase_metrics.groupby(
            [
                "config_name",
                "detector_mode",
                "rule_mode",
                "feature_set",
                "min_alert_windows",
                "cooldown_windows",
                "phase",
                "phase_kind",
            ]
        )[["recall", "false_positive_rate", "accuracy", "precision", "f1"]]
        .mean()
        .reset_index()
    )

    ablation_metrics = pd.DataFrame(ablation_records)
    final_comparison = summary_metrics[summary_metrics["feature_set"] == "full"][
        [
            "config_name",
            "accuracy_mean",
            "precision_mean",
            "recall_mean",
            "false_positive_rate_mean",
            "f1_mean",
            "firmware_burst_fpr_mean",
            "udp_recall_mean",
            "syn_recall_mean",
        ]
    ].sort_values("config_name")

    per_run_metrics.to_csv(os.path.join(args.results_dir, "per_run_metrics.csv"), index=False)
    summary_metrics.to_csv(os.path.join(args.results_dir, "summary_metrics.csv"), index=False)
    phase_summary.to_csv(os.path.join(args.results_dir, "phase_metrics.csv"), index=False)
    ablation_metrics.to_csv(os.path.join(args.results_dir, "ablation_metrics.csv"), index=False)
    final_comparison.to_csv(os.path.join(args.results_dir, "final_comparison.csv"), index=False)

    plot_detector_comparison(summary_metrics, os.path.join(args.results_dir, "detector_comparison.png"))
    plot_phase_metrics(phase_summary, os.path.join(args.results_dir, "phase_metrics.png"))
    plot_multi_seed_summary(summary_metrics, os.path.join(args.results_dir, "multi_seed_summary.png"))
    plot_ablation_comparison(ablation_metrics, os.path.join(args.results_dir, "ablation_comparison.png"))
    plot_final_tradeoff_bars(summary_metrics, os.path.join(args.results_dir, "final_tradeoff_bars.png"))
    plot_final_phase_breakdown(summary_metrics, os.path.join(args.results_dir, "final_phase_breakdown.png"))


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
    parser.add_argument("--detector-mode", default="adaptive", choices=["adaptive", "fixed", "hybrid"], help="Detector mode")
    parser.add_argument("--rule-mode", default="protocol", choices=["protocol", "kofn"], help="Decision rule mode")
    parser.add_argument("--calibration-seconds", type=int, default=60, help="Calibration period for fixed thresholds")
    parser.add_argument("--kofn-k", type=int, default=2, help="Number of threshold hits required for kofn mode")
    parser.add_argument("--min-alert-windows", type=int, default=1, help="Consecutive windows required before alerting")
    parser.add_argument("--cooldown-windows", type=int, default=0, help="Windows to keep alert active after signal drops")
    parser.add_argument("--seed", type=int, default=1337, help="Base random seed")
    parser.add_argument("--run-experiments", action="store_true", help="Run multi-seed comparison experiments")
    parser.add_argument("--seeds", type=int, default=10, help="Number of sequential seeds to evaluate")
    parser.add_argument("--ablation-detector-mode", default="adaptive", choices=["adaptive", "fixed", "hybrid"], help="Detector mode for ablation experiments")
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
        min_alert_windows=args.min_alert_windows,
        cooldown_windows=args.cooldown_windows,
    )

    if args.run_experiments:
        run_experiments(args)


if __name__ == "__main__":
    main()
