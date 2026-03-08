# IoT DoS Detection (Adaptive Thresholds)

This project generates synthetic IoT traffic (normal + benign burst + UDP flood + SYN flood),
extracts simple per-second features, detects anomalies with adaptive thresholds, and evaluates
performance including a threshold sweep.

## Pipeline
1. Generate PCAP + ground-truth labels
2. Extract features per 1s window
3. Detect using rolling baseline thresholds
4. Simulate mitigation (log actions)
5. Evaluate metrics and plot time series + ROC-like sweep

## Quick Start
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python main.py --generate --sweep
```

## Outputs
- `results/features.csv`
- `results/detections.csv`
- `results/labels.csv`
- `results/metrics.txt`
- `results/mitigation.log`
- `results/timeseries.png`
- `results/roc_like.png` (if `--sweep`)

## Notes
- Adaptive thresholds use the last `baseline_window` seconds.
- The benign firmware burst is labeled as normal to test false positives.
