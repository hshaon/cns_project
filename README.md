# IoT DoS Detection (Progress Report 2)

This project generates synthetic IoT traffic, extracts per-second features, compares rule-based
detectors, and produces report-ready experiment outputs for the final project stage.

## Pipeline
1. Generate labeled traffic with normal, benign burst, UDP flood, and SYN flood phases.
2. Extract aligned per-second features, including zero-traffic windows.
3. Detect anomalies with adaptive, fixed, or hybrid thresholds.
4. Smooth alerts with configurable persistence and cooldown windows.
5. Compare detector configurations across multiple seeds.
6. Run feature ablation experiments.
7. Produce metrics, phase summaries, and report figures.

## Quick Start
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python main.py --generate --sweep --run-experiments
```

## Key CLI Options
- `--detector-mode adaptive|fixed|hybrid`
- `--rule-mode protocol|kofn`
- `--calibration-seconds 60`
- `--min-alert-windows 2`
- `--cooldown-windows 0`
- `--run-experiments`
- `--seeds 10`
- `--ablation-detector-mode adaptive`

## Main Outputs
- `results/features.csv`
- `results/detections.csv`
- `results/labels.csv`
- `results/thresholds.csv`
- `results/metrics.txt`
- `results/per_run_metrics.csv`
- `results/summary_metrics.csv`
- `results/phase_metrics.csv`
- `results/ablation_metrics.csv`
- `results/final_comparison.csv`

## Figures
- `results/timeseries.png`
- `results/roc_like.png`
- `results/detector_comparison.png`
- `results/phase_metrics.png`
- `results/multi_seed_summary.png`
- `results/ablation_comparison.png`

## Notes
- The canonical timeline comes from `labels.csv`, so `labels.csv`, `features.csv`, and `detections.csv`
  are aligned row-for-row.
- The final detector comparison uses `adaptive`, `fixed`, `hybrid`, and `hybrid_persistent` configurations.
- `hybrid` uses adaptive thresholds for UDP-style features and fixed thresholds for SYN-style features.
- Persistence is controlled by `min_alert_windows`, while `cooldown_windows` keeps an alert active briefly after detection.
- The benign firmware burst remains labeled as normal so false positives are visible in evaluation.
