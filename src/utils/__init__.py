"""Utility functions, configuration, device detection, and visualization tools."""
from src.utils.config import ExperimentConfig
from src.utils.device import get_device_info, print_device_banner, resolve_device
from src.utils.logger import setup_experiment_logger
from src.utils.visualizer import (
    generate_all_visualizations,
    plot_accuracy_matrix_heatmap,
    plot_continual_vs_baseline_barchart,
    plot_forgetting_decay_curves,
)

__all__ = [
    "ExperimentConfig",
    "get_device_info",
    "print_device_banner",
    "resolve_device",
    "setup_experiment_logger",
    "generate_all_visualizations",
    "plot_accuracy_matrix_heatmap",
    "plot_continual_vs_baseline_barchart",
    "plot_forgetting_decay_curves",
]

