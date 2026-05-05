# IoT DoS Detection (Progress Report 2)

This project generates synthetic IoT traffic, extracts per-second features, compares rule-based
detectors, and produces report-ready experiment outputs for Progress Report 2.

## Pipeline
1. Generate labeled traffic with normal, benign burst, UDP flood, and SYN flood phases.
2. Extract aligned per-second features, including zero-traffic windows.
3. Detect anomalies with either adaptive or fixed thresholds.
4. Compare detector modes across multiple seeds.
5. Run feature ablation experiments.
6. Produce metrics, phase summaries, and report figures.

## Quick Start
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python main.py --generate --sweep --run-experiments
```

## Key CLI Options
- `--detector-mode adaptive|fixed`
- `--rule-mode protocol|kofn`
- `--calibration-seconds 60`
- `--run-experiments`
- `--seeds 10`
- `--experiment-detector-modes adaptive,fixed`
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
- The default detector comparison uses `adaptive` and `fixed` threshold modes with the `protocol` rule.
- The benign firmware burst remains labeled as normal so false positives are visible in evaluation.
