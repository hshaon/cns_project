from typing import List

import matplotlib.pyplot as plt
import pandas as pd


FEATURES = [
    "packets_per_sec",
    "bytes_per_sec",
    "syn_per_sec",
    "unique_dst_ports",
    "unique_dst_ips",
]


def plot_timeseries(features: pd.DataFrame, truth: pd.Series, output_path: str) -> None:
    fig, axes = plt.subplots(len(FEATURES) + 1, 1, figsize=(10, 12), sharex=True)

    x = features["window_start"]
    for i, feature in enumerate(FEATURES):
        axes[i].plot(x, features[feature], label=feature)
        axes[i].set_ylabel(feature)
        axes[i].grid(True, alpha=0.3)

    axes[-1].step(x, truth, where="post", label="ground_truth", color="red")
    axes[-1].set_ylabel("attack")
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_ylim(-0.1, 1.1)
    axes[-1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc(fprs, tprs, output_path: str, mults: List[float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fprs, tprs, marker="o")
    for i, m in enumerate(mults):
        ax.annotate(f"{m:.2f}", (fprs[i], tprs[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Threshold Sweep (ROC-like)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
