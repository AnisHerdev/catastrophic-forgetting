"""
Visualization Suite for Continual Learning and Catastrophic Forgetting.
Generates publication-quality heatmaps, accuracy degradation curves, and baseline comparison charts.
"""

import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    title_suffix: str = "",
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

    title_main = f"Continual Learning Accuracy Matrix $R_{{i,j}}$ (%){title_suffix}"
    ax.set_title(f"{title_main}\n(Row: Model State, Column: Evaluated Phase)", fontsize=14, weight="bold", pad=15)
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
    title_suffix: str = "",
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

    title_main = f"Continual Learning: Accuracy Decay Across Sequential Phases{title_suffix}"
    ax.set_title(title_main, fontsize=14, weight="bold", pad=15)
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
    title_suffix: str = "",
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

    title_main = f"Performance Comparison: Initial vs Retained vs Isolated Baselines{title_suffix}"
    ax.set_title(title_main, fontsize=14, weight="bold", pad=15)
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


def plot_naive_vs_replay_comparison(
    naive_csv_path: str,
    replay_tracker: ContinualMetricsTracker,
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Generates a dual-panel publication-grade comparison contrasting
    unmitigated naive continual learning with experience replay.
    """
    if not os.path.exists(naive_csv_path):
        return None

    try:
        df_naive = pd.read_csv(naive_csv_path, index_col=0)
    except Exception:
        return None

    setup_plotting_style()
    num_phases = replay_tracker.num_phases
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Subplot 1: Phase 1 Accuracy Decay Trajectory across Stages 1..4
    stages = list(range(1, num_phases + 1))
    naive_p1_decay = []
    for i in range(1, num_phases + 1):
        row_name = f"After Training Phase {i}"
        val = df_naive.loc[row_name, "Eval Phase 1"] if row_name in df_naive.index and "Eval Phase 1" in df_naive.columns else np.nan
        naive_p1_decay.append(float(val) if pd.notna(val) else 0.0)

    replay_p1_decay = [replay_tracker.accuracy_matrix[i - 1, 0] for i in stages]

    ax1.plot(stages, naive_p1_decay, "o--", color="#d95f02", linewidth=2.5, markersize=8, label="Naive Continual (Unmitigated)")
    ax1.plot(stages, replay_p1_decay, "s-", color="#1b9e77", linewidth=2.5, markersize=8, label="Continual + Replay Buffer")

    ax1.set_title("Phase 1 Retention: Naive vs Replay Buffer", fontsize=13, weight="bold", pad=12)
    ax1.set_xlabel("Continual Training Stage", fontsize=11, labelpad=8)
    ax1.set_ylabel("Phase 1 Test Accuracy (%)", fontsize=11, labelpad=8)
    ax1.set_xticks(stages)
    ax1.set_xticklabels([f"Stage {i}" for i in stages])
    ax1.set_ylim(-2, 105)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    # Subplot 2: Cumulative Accuracy on All Seen Classes
    naive_cum = []
    for i in range(1, num_phases + 1):
        row_name = f"After Training Phase {i}"
        val = df_naive.loc[row_name, "Cumulative Test Acc"] if row_name in df_naive.index and "Cumulative Test Acc" in df_naive.columns else np.nan
        naive_cum.append(float(val) if pd.notna(val) else 0.0)

    replay_cum = [replay_tracker.cumulative_accuracies.get(i, np.nan) for i in stages]

    x = np.arange(len(stages))
    width = 0.35
    ax2.bar(x - width/2, naive_cum, width, label="Naive Continual", color="#e7298a")
    ax2.bar(x + width/2, replay_cum, width, label="Continual + Replay Buffer", color="#386cb0")

    ax2.set_title("Cumulative Lifetime Accuracy (All Seen Classes)", fontsize=13, weight="bold", pad=12)
    ax2.set_xlabel("Continual Training Stage", fontsize=11, labelpad=8)
    ax2.set_ylabel("Accuracy (%)", fontsize=11, labelpad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Stage {i} ({i*25} cls)" for i in stages])
    ax2.set_ylim(0, 105)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Visualization] Saved Naive vs Replay Comparison Chart to: {save_path}")

    return fig


def generate_all_visualizations(
    tracker: ContinualMetricsTracker,
    output_dir: str = "./results",
    suffix: str = "",
) -> None:
    """Generates and saves all plots and metrics to the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    title_suffix = " (Replay Buffer)" if "replay" in suffix else ""

    plot_accuracy_matrix_heatmap(
        tracker,
        save_path=os.path.join(output_dir, f"accuracy_matrix_heatmap{suffix}.png"),
        title_suffix=title_suffix,
    )
    plot_forgetting_decay_curves(
        tracker,
        save_path=os.path.join(output_dir, f"accuracy_decay_curves{suffix}.png"),
        title_suffix=title_suffix,
    )
    plot_continual_vs_baseline_barchart(
        tracker,
        save_path=os.path.join(output_dir, f"continual_vs_baseline{suffix}.png"),
        title_suffix=title_suffix,
    )
    plt.close("all")
