import argparse
import os

import numpy as np
import pandas as pd

from detector import detect
from evaluator import evaluate, sweep_thresholds
from feature_extractor import extract_features
from mitigation import simulate_mitigation
from traffic_generator import generate_pcap
from visualizer import plot_timeseries, plot_roc


DEFAULT_PCAP = "pcaps/iot_traffic.pcap"
DEFAULT_LABELS = "results/labels.csv"


def run_all(args: argparse.Namespace) -> None:
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.pcap), exist_ok=True)

    if args.generate:
        generate_pcap(args.pcap, args.labels)

    features = extract_features(args.pcap, window_s=args.window)
    features.to_csv(os.path.join(args.results_dir, "features.csv"), index=False)

    labels = pd.read_csv(args.labels)
    merged = features.merge(labels, on="window_start", how="left").fillna({"label": 0})

    detections, _ = detect(
        merged,
        baseline_window=args.baseline_window,
        threshold_mult=args.threshold_mult,
        min_baseline=args.min_baseline,
    )

    detections.to_csv(os.path.join(args.results_dir, "detections.csv"), index=False)

    metrics = evaluate(detections["prediction"], merged["label"])
    with open(os.path.join(args.results_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    simulate_mitigation(detections, os.path.join(args.results_dir, "mitigation.log"))

    plot_timeseries(merged, merged["label"], os.path.join(args.results_dir, "timeseries.png"))

    if args.sweep:
        mults = np.arange(args.sweep_min, args.sweep_max + 1e-9, args.sweep_step)
        fprs, tprs, mult_vals = sweep_thresholds(
            merged,
            merged["label"],
            detect,
            mults,
            baseline_window=args.baseline_window,
            min_baseline=args.min_baseline,
        )
        plot_roc(fprs, tprs, os.path.join(args.results_dir, "roc_like.png"), list(mult_vals))



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
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep")
    parser.add_argument("--sweep-min", type=float, default=1.0, help="Sweep min multiplier")
    parser.add_argument("--sweep-max", type=float, default=5.0, help="Sweep max multiplier")
    parser.add_argument("--sweep-step", type=float, default=0.5, help="Sweep step")
    return parser.parse_args()


if __name__ == "__main__":
    run_all(parse_args())
