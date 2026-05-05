from typing import Iterable, List

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

    x_axis = features["window_start"]
    for index, feature in enumerate(FEATURES):
        axes[index].plot(x_axis, features[feature], label=feature)
        axes[index].set_ylabel(feature)
        axes[index].grid(True, alpha=0.3)

    axes[-1].step(x_axis, truth, where="post", label="ground_truth", color="red")
    axes[-1].set_ylabel("attack")
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_ylim(-0.1, 1.1)
    axes[-1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc(fprs, tprs, output_path: str, mults: List[float]) -> None:
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fprs, tprs, marker="o")
    for index, mult in enumerate(mults):
        axis.annotate(f"{mult:.2f}", (fprs[index], tprs[index]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("Threshold Sweep (ROC-like)")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _bar_plot(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    output_path: str,
    title: str,
    ylabel: str,
    yerr_col: str = "",
) -> None:
    pivot = frame.pivot(index=x_col, columns=hue_col, values=y_col).fillna(0.0)
    errors = None
    if yerr_col:
        errors = frame.pivot(index=x_col, columns=hue_col, values=yerr_col).fillna(0.0)

    axis = pivot.plot(
        kind="bar",
        figsize=(9, 5),
        yerr=errors,
        capsize=4 if errors is not None else 0,
    )
    axis.set_title(title)
    axis.set_xlabel(x_col.replace("_", " ").title())
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_detector_comparison(summary_metrics: pd.DataFrame, output_path: str) -> None:
    frame = summary_metrics[summary_metrics["feature_set"] == "full"]
    _bar_plot(
        frame=frame,
        x_col="detector_mode",
        y_col="recall_mean",
        hue_col="rule_mode",
        output_path=output_path,
        title="Detector Comparison",
        ylabel="Mean Recall",
        yerr_col="recall_std",
    )


def plot_phase_metrics(phase_metrics_frame: pd.DataFrame, output_path: str) -> None:
    frame = phase_metrics_frame.copy()
    preferred = frame[
        (frame["feature_set"] == "full")
        & (frame["detector_mode"] == "adaptive")
        & (frame["rule_mode"] == "protocol")
    ]
    if not preferred.empty:
        frame = preferred
    frame["metric_value"] = frame.apply(
        lambda row: row["recall"] if row["phase_kind"] == "attack" else row["false_positive_rate"],
        axis=1,
    )
    frame["metric_name"] = frame["phase_kind"].map({"attack": "Recall", "benign": "False Positive Rate"})
    _bar_plot(
        frame=frame,
        x_col="phase",
        y_col="metric_value",
        hue_col="metric_name",
        output_path=output_path,
        title="Per-Phase Detection Behavior",
        ylabel="Rate",
    )


def plot_multi_seed_summary(summary_metrics: pd.DataFrame, output_path: str) -> None:
    frame = summary_metrics[summary_metrics["feature_set"] == "full"]
    melted = frame.melt(
        id_vars=["detector_mode", "rule_mode"],
        value_vars=["accuracy_mean", "precision_mean", "recall_mean", "false_positive_rate_mean", "f1_mean"],
        var_name="metric",
        value_name="value",
    )
    melted["metric"] = melted["metric"].str.replace("_mean", "", regex=False)
    melted["config"] = melted["detector_mode"] + " / " + melted["rule_mode"]

    axis = melted.pivot(index="metric", columns="config", values="value").plot(kind="bar", figsize=(10, 5))
    axis.set_title("Multi-Seed Metric Summary")
    axis.set_xlabel("Metric")
    axis.set_ylabel("Mean Value")
    axis.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ablation_comparison(ablation_metrics: pd.DataFrame, output_path: str) -> None:
    frame = ablation_metrics.groupby("feature_set")[["recall", "false_positive_rate"]].mean().reset_index()
    melted = frame.melt(id_vars=["feature_set"], value_vars=["recall", "false_positive_rate"], var_name="metric", value_name="value")
    _bar_plot(
        frame=melted,
        x_col="feature_set",
        y_col="value",
        hue_col="metric",
        output_path=output_path,
        title="Feature Ablation Comparison",
        ylabel="Mean Value",
    )
