"""
Visualization Suite for Continual Learning and Catastrophic Forgetting.
Generates publication-quality heatmaps, accuracy degradation curves, and baseline comparison charts.
"""

import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.evaluation.metrics import ContinualMetricsTracker


def setup_plotting_style() -> None:
    """Configures clean, modern aesthetic for matplotlib & seaborn plots."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.0


def plot_accuracy_matrix_heatmap(
    tracker: ContinualMetricsTracker,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots an annotated heatmap of the continual learning accuracy matrix R_{i, j}.
    """
    setup_plotting_style()
    num_phases = tracker.num_phases
    matrix = tracker.accuracy_matrix

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Mask upper triangle (future phases not yet trained)
    mask = np.isnan(matrix)

    # Use a refined colormap
    cmap = sns.color_palette("mako", as_cmap=True)

    sns.heatmap(
        matrix,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        vmin=0.0,
        vmax=100.0,
        cbar_kws={"label": "Accuracy (%)"},
        xticklabels=[f"Phase {j+1}" for j in range(num_phases)],
        yticklabels=[f"Trained Phase {i+1}" for i in range(num_phases)],
        linewidths=1.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )

    ax.set_title("Continual Learning Accuracy Matrix $R_{i,j}$ (%)\n(Row: Model State, Column: Evaluated Phase)", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Evaluated Phase Test Set", fontsize=12, labelpad=10)
    ax.set_ylabel("After Completing Training On", fontsize=12, labelpad=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Visualization] Saved Accuracy Matrix Heatmap to: {save_path}")

    return fig


def plot_forgetting_decay_curves(
    tracker: ContinualMetricsTracker,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots the accuracy trajectory of each phase across subsequent training steps,
    illustrating the severity of catastrophic forgetting over time.
    """
    setup_plotting_style()
    num_phases = tracker.num_phases
    matrix = tracker.accuracy_matrix

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    palette = sns.color_palette("tab10", num_phases)
    markers = ["o", "s", "^", "D"]

    for phase_idx in range(num_phases):
        x_steps = list(range(phase_idx + 1, num_phases + 1))
        y_accs = [matrix[step - 1, phase_idx] for step in x_steps]

        if any(not np.isnan(y) for y in y_accs):
            ax.plot(
                x_steps,
                y_accs,
                marker=markers[phase_idx % len(markers)],
                markersize=9,
                linewidth=2.5,
                label=f"Phase {phase_idx+1} Test Set",
                color=palette[phase_idx],
            )

    ax.set_title("Catastrophic Forgetting: Accuracy Decay Across Sequential Phases", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Continual Training Stage (After Phase N)", fontsize=12, labelpad=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12, labelpad=10)
    ax.set_xticks(range(1, num_phases + 1))
    ax.set_xticklabels([f"Stage {i} (P{i})" for i in range(1, num_phases + 1)])
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11, loc="best")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Visualization] Saved Accuracy Decay Curves to: {save_path}")

    return fig


def plot_continual_vs_baseline_barchart(
    tracker: ContinualMetricsTracker,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots a grouped bar chart comparing Initial Accuracy (R_{j,j}) vs Retained Final Accuracy (R_{4,j})
    vs Isolated Baseline for each phase.
    """
    setup_plotting_style()
    num_phases = tracker.num_phases
    matrix = tracker.accuracy_matrix

    phases = list(range(1, num_phases + 1))
    initial_accs = [matrix[j - 1, j - 1] for j in phases]
    final_accs = [matrix[num_phases - 1, j - 1] for j in phases]
    has_baselines = bool(tracker.baseline_accuracies)

    x = np.arange(len(phases))
    width = 0.25 if has_baselines else 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    rects1 = ax.bar(x - width if has_baselines else x - width/2, initial_accs, width, label="Initial Accuracy ($R_{j,j}$)", color="#2b5c8f")
    rects2 = ax.bar(x if has_baselines else x + width/2, final_accs, width, label="Final Retained Accuracy ($R_{N,j}$)", color="#d95f02")

    if has_baselines:
        baseline_vals = [tracker.baseline_accuracies.get(j, np.nan) for j in phases]
        rects3 = ax.bar(x + width, baseline_vals, width, label="Isolated Single-Phase Baseline", color="#7570b3")

    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    weight="bold",
                )

    autolabel(rects1)
    autolabel(rects2)
    if has_baselines:
        autolabel(rects3)

    ax.set_title("Performance Comparison: Initial vs Retained (Continual) vs Isolated Baselines", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Phase Task", fontsize=12, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Phase {j}" for j in phases])
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11, loc="upper right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Visualization] Saved Baseline Comparison Bar Chart to: {save_path}")

    return fig


def generate_all_visualizations(
    tracker: ContinualMetricsTracker,
    output_dir: str = "./results",
) -> None:
    """Generates and saves all plots and metrics to the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    plot_accuracy_matrix_heatmap(tracker, save_path=os.path.join(output_dir, "accuracy_matrix_heatmap.png"))
    plot_forgetting_decay_curves(tracker, save_path=os.path.join(output_dir, "accuracy_decay_curves.png"))
    plot_continual_vs_baseline_barchart(tracker, save_path=os.path.join(output_dir, "continual_vs_baseline.png"))
    plt.close("all")
