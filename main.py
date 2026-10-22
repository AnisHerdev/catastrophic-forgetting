"""
Main CLI Entrypoint: Quantifying Catastrophic Forgetting in Class-Incremental Learning
with VGG16-Adapted Networks on CIFAR-100.
"""

import argparse
import os
import random
import numpy as np
import torch

from src.training.trainer import ContinualLearningPipeline
from src.utils.config import ExperimentConfig
from src.utils.device import print_device_banner, resolve_device


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_banner() -> None:
    banner = """
================================================================================
  QUANTIFYING CATASTROPHIC FORGETTING IN CLASS-INCREMENTAL CONTINUAL LEARNING
                   VGG16-Adapted Network on CIFAR-100 (4 Phases)
================================================================================
    """
    print(banner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantifying Catastrophic Forgetting in Class-Incremental Learning on CIFAR-100."
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs per continual phase (default: 15)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size for DataLoader (default: 64)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Initial learning rate for Adam optimizer (default: 0.001)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for regularization (default: 1e-4)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout probability in classifier head (default: 0.5)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Directory for CIFAR-100 dataset (default: ./data)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results", help="Directory to save CSV results and plots (default: ./results)"
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default="./checkpoints", help="Directory to save model checkpoints (default: ./checkpoints)"
    )
    parser.add_argument(
        "--no-baselines", action="store_true", help="Skip running isolated single-phase baseline models"
    )
    parser.add_argument(
        "--baseline-epochs", type=int, default=15, help="Epochs for isolated baseline models (default: 15)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Hardware compute device to run on ('auto', 'cuda', 'cpu', 'mps', default: auto-detect)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="Run a fast 1-epoch sanity check across all 4 phases"
    )
    return parser.parse_args()


def main() -> None:
    print_banner()
    args = parse_args()

    # Detect and resolve hardware compute device (CUDA GPU -> Apple Silicon MPS -> CPU)
    device, device_info = resolve_device(args.device)
    print_device_banner(device_info)

    # Set seed
    set_seed(args.seed)

    # If quick-test flag is passed, reduce epochs for smoke testing
    epochs_per_phase = 1 if args.quick_test else args.epochs
    baseline_epochs = 1 if args.quick_test else args.baseline_epochs

    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoints_dir=args.checkpoints_dir,
        pretrained_backbone=True,
        dropout_rate=args.dropout,
        num_phases=4,
        superclasses_per_phase=5,
        classes_per_phase=25,
        epochs_per_phase=epochs_per_phase,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        run_isolated_baselines=not args.no_baselines,
        baseline_epochs=baseline_epochs,
        device=str(device),
        seed=args.seed,
    )

    print(f"[Configuration]")
    print(f"  - Active Hardware Device : {device_info['device_type'].upper()} ({device})")
    print(f"  - Continual Phases       : {config.num_phases} (25 classes each -> 100 total)")
    print(f"  - Epochs Per Phase       : {config.epochs_per_phase}")
    print(f"  - Batch Size             : {config.batch_size}")
    print(f"  - Learning Rate          : {config.learning_rate}")
    print(f"  - Run Isolated Baselines : {config.run_isolated_baselines} ({config.baseline_epochs} epochs each)")
    print(f"  - Output Directory       : {config.output_dir}")
    print(f"  - Checkpoints Directory  : {config.checkpoints_dir}")

    # Launch Pipeline
    pipeline = ContinualLearningPipeline(config)
    pipeline.run_full_experiment()

    print("\n[Complete] Experiment finished successfully!")


if __name__ == "__main__":
    main()
