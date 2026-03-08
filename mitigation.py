from typing import List

import pandas as pd


def simulate_mitigation(detections: pd.DataFrame, output_path: str) -> None:
    alerts: List[str] = []
    for _, row in detections.iterrows():
        if int(row["prediction"]) == 1:
            alerts.append(
                f"window {int(row['window_start'])}: drop/rate-limit source (simulated)"
            )

    with open(output_path, "w", encoding="utf-8") as f:
        if not alerts:
            f.write("No mitigation actions triggered.\n")
            return
        for line in alerts:
            f.write(line + "\n")
