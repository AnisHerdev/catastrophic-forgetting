"""
Continual Learning Trainer and Experiment Execution Engine.
Orchestrates multi-phase class-incremental training, dynamic head expansion,
isolated single-phase baselines, and evaluation matrix tracking.
"""

import os
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.cifar100_split import CIFAR100ContinualDataModule
from src.data.replay_buffer import ReplayBuffer
from src.evaluation.metrics import ContinualMetricsTracker, evaluate_accuracy
from src.models.vgg16_adapted import VGG16Adapted
from src.utils.config import ExperimentConfig
from src.utils.device import print_device_banner, resolve_device
from src.utils.seed import set_seed
from src.utils.visualizer import generate_all_visualizations, plot_naive_vs_replay_comparison


def train_single_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Training",
) -> Tuple[float, float]:
    """
    Executes a single training epoch and returns (epoch_accuracy, epoch_loss).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=desc, leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        current_acc = (correct / total) * 100.0 if total > 0 else 0.0
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.2f}%"})

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total) * 100.0 if total > 0 else 0.0
    return epoch_acc, epoch_loss


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    phase_label: str = "Phase",
) -> nn.Module:
    """
    Trains a model for a given number of epochs using the Adam optimizer.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        acc, loss = train_single_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            desc=f"[{phase_label}] Epoch {epoch}/{epochs}",
        )
        elapsed = time.time() - start_time
        print(f"[{phase_label}] Epoch {epoch:02d}/{epochs:02d} | Loss: {loss:.4f} | Train Acc: {acc:.2f}% | Time: {elapsed:.1f}s")

    return model


class ContinualLearningPipeline:
    """
    Manages the complete 4-phase continual class-incremental learning workflow:
    Phase 1 (25 classes) -> Phase 2 (50 classes) -> Phase 3 (75 classes) -> Phase 4 (100 classes).
    Supports Experience Replay Buffer and baseline caching.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device, self.device_info = resolve_device(config.device)
        print(f"[Pipeline] Active compute device: {self.device}")

        # Setup Data Module
        print("\n[Data] Initializing CIFAR-100 Continual Data Module...")
        self.data_module = CIFAR100ContinualDataModule(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            superclasses_per_phase=config.superclasses_per_phase,
        )
        self.data_module.print_phase_summary()

        # Setup Metrics Tracker
        self.tracker = ContinualMetricsTracker(num_phases=config.num_phases)

        # Ensure output and checkpoint directories exist
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoints_dir, exist_ok=True)

    def run_isolated_baselines(self) -> None:
        """
        Trains independent, isolated models for each phase to establish peak single-task baseline bounds.
        Reuses cached baseline checkpoints if available.
        """
        print("\n" + "=" * 80)
        print("RUNNING ISOLATED SINGLE-PHASE BASELINES (Upper Bounds)")
        print("=" * 80)

        for phase in range(1, self.config.num_phases + 1):
            phase_seed = self.config.seed if phase == 1 else (self.config.seed + phase)
            set_seed(phase_seed)

            ckpt_path = os.path.join(self.config.checkpoints_dir, f"baseline_phase_{phase}.pt")

            # Check if cached baseline checkpoint exists
            if os.path.exists(ckpt_path) and not self.config.force_baselines:
                print(f"\n---> [Baseline Cache] Found cached model for Phase {phase}: {ckpt_path}")
                baseline_model = VGG16Adapted(
                    num_classes=self.config.classes_per_phase,
                    pretrained=self.config.pretrained_backbone,
                    dropout_rate=self.config.dropout_rate,
                ).to(self.device)
                baseline_model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
                _, test_loader = self.data_module.get_phase_dataloaders(phase, mode="isolated")
                base_acc, base_loss = evaluate_accuracy(baseline_model, test_loader, self.device)
                self.tracker.record_baseline(phase, base_acc)
                print(f"[Baseline Result] Phase {phase} Cached Test Accuracy: {base_acc:.2f}% (Loss: {base_loss:.4f})")
                continue

            print(f"\n---> Training Isolated Baseline Model for Phase {phase} ({self.config.classes_per_phase} classes)...")

            # Initialize fresh VGG16 model with 25 classes
            baseline_model = VGG16Adapted(
                num_classes=self.config.classes_per_phase,
                pretrained=self.config.pretrained_backbone,
                dropout_rate=self.config.dropout_rate,
            ).to(self.device)

            train_loader, test_loader = self.data_module.get_phase_dataloaders(
                phase, mode="isolated", seed=phase_seed
            )

            train_model(
                model=baseline_model,
                train_loader=train_loader,
                epochs=self.config.baseline_epochs,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                device=self.device,
                phase_label=f"Baseline P{phase}",
            )

            # Evaluate baseline model
            base_acc, base_loss = evaluate_accuracy(baseline_model, test_loader, self.device)
            self.tracker.record_baseline(phase, base_acc)
            print(f"[Baseline Result] Phase {phase} Isolated Test Accuracy: {base_acc:.2f}% (Loss: {base_loss:.4f})")

            # Save baseline checkpoint
            torch.save(baseline_model.state_dict(), ckpt_path)

    def run_continual_learning(self) -> None:
        """
        Executes sequential continual learning across the 4 phases with dynamic head expansion,
        optionally integrating an Experience Replay Buffer.
        """
        print("\n" + "=" * 80)
        mode_title = "CLASS-INCREMENTAL CONTINUAL LEARNING (WITH REPLAY BUFFER)" if self.config.use_replay else "CLASS-INCREMENTAL CONTINUAL LEARNING PIPELINE"
        print(f"STARTING {mode_title}")
        print("=" * 80)

        # Initialize Replay Buffer if enabled
        replay_buffer: Optional[ReplayBuffer] = None
        if self.config.use_replay:
            replay_buffer = ReplayBuffer(
                base_dataset=self.data_module.raw_train_dataset,
                class_mapping=self.data_module.global_class_mapping,
                samples_per_class=self.config.replay_samples_per_class,
            )
            print(f"[Replay Buffer] Initialized Experience Replay Memory ({self.config.replay_samples_per_class} exemplars/class)")

        # Reset RNG state to self.config.seed so Continual Phase 1 begins with
        # the EXACT same initial weights, data shuffling, and dropout as Baseline Phase 1.
        set_seed(self.config.seed)

        # Initialize Continual Model with initial 25 classes
        continual_model = VGG16Adapted(
            num_classes=self.config.classes_per_phase,
            pretrained=self.config.pretrained_backbone,
            dropout_rate=self.config.dropout_rate,
        ).to(self.device)

        # Iterate through continual phases
        for current_phase in range(1, self.config.num_phases + 1):
            expected_num_classes = current_phase * self.config.classes_per_phase

            print(f"\n{'#' * 80}")
            print(f" CONTINUAL PHASE {current_phase}/{self.config.num_phases} : Total Seen Classes = {expected_num_classes}")
            print(f"{'#' * 80}")

            # Expand classification head if past Phase 1
            if current_phase > 1:
                print(f"[Model Head] Expanding output layer: {continual_model.num_classes} -> {expected_num_classes} classes...")
                continual_model.expand_classes(expected_num_classes)

            # Get current phase training data (interleave replay buffer for phase > 1 if active)
            phase_loader_seed = self.config.seed if current_phase == 1 else None
            active_replay = replay_buffer if (self.config.use_replay and current_phase > 1) else None

            train_loader, _ = self.data_module.get_phase_dataloaders(
                current_phase, mode="continual", seed=phase_loader_seed, replay_dataset=active_replay
            )

            if active_replay is not None and len(active_replay) > 0:
                print(f"[Replay Buffer] Interleaving {len(active_replay)} exemplars with Phase {current_phase} data (Total training samples: {len(train_loader.dataset)})")

            # Train on current phase
            train_model(
                model=continual_model,
                train_loader=train_loader,
                epochs=self.config.epochs_per_phase,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                device=self.device,
                phase_label=f"Continual P{current_phase}",
            )

            # Save checkpoint after this phase
            suffix = "_replay_buffer" if self.config.use_replay else ""
            ckpt_path = os.path.join(self.config.checkpoints_dir, f"continual_model_phase_{current_phase}{suffix}.pt")
            torch.save(continual_model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model checkpoint to {ckpt_path}")

            # Add completed phase exemplars into replay buffer
            if self.config.use_replay and replay_buffer is not None:
                added = replay_buffer.add_phase_classes(
                    self.data_module.phase_fine_classes[current_phase], seed=self.config.seed
                )
                print(f"[Replay Buffer] Added {added} exemplars from Phase {current_phase}. Total buffer size: {len(replay_buffer)} samples.")

            # Comprehensive Evaluation: Evaluate on all seen phase test sets (1 to current_phase)
            print(f"\n[Evaluation after Phase {current_phase}] Evaluating model on all seen test sets:")
            for eval_p in range(1, current_phase + 1):
                t_loader = self.data_module.get_phase_test_loader(eval_p)
                eval_acc, eval_loss = evaluate_accuracy(continual_model, t_loader, self.device)
                self.tracker.record_accuracy(trained_phase=current_phase, evaluated_phase=eval_p, accuracy=eval_acc)
                
                # Check for forgetting compared to peak initial accuracy
                initial_acc = self.tracker.accuracy_matrix[eval_p - 1, eval_p - 1]
                delta_str = f" (Δ: {eval_acc - initial_acc:+.2f}%)" if eval_p < current_phase else ""
                print(f"  -> Accuracy on Phase {eval_p} Test Set: {eval_acc:.2f}%{delta_str}")

            # Evaluate on cumulative test set of all seen classes
            cum_loader = self.data_module.get_cumulative_test_loader(up_to_phase=current_phase)
            cum_acc, cum_loss = evaluate_accuracy(continual_model, cum_loader, self.device)
            self.tracker.record_cumulative(current_phase, cum_acc)
            print(f"  -> Cumulative Test Accuracy (All {expected_num_classes} Seen Classes): {cum_acc:.2f}%\n")

    def run_full_experiment(self) -> ContinualMetricsTracker:
        """Runs baseline experiments (optional), continual learning, saves CSV and generates plots."""
        if self.config.run_isolated_baselines:
            self.run_isolated_baselines()

        self.run_continual_learning()

        # Display Summary Table
        self.tracker.print_summary_table()

        suffix = "_replay_buffer" if self.config.use_replay else ""

        # Save Metrics CSV
        csv_path = os.path.join(self.config.output_dir, f"continual_forgetting_results{suffix}.csv")
        self.tracker.save_csv(csv_path)
        print(f"[Results] Saved metrics to CSV: {csv_path}")

        # Generate Visualizations
        print("\n[Visualizations] Generating analysis plots...")
        generate_all_visualizations(self.tracker, output_dir=self.config.output_dir, suffix=suffix)

        # If replay run, generate side-by-side comparison with baseline run if available
        if self.config.use_replay:
            naive_csv = os.path.join(self.config.output_dir, "continual_forgetting_results.csv")
            comp_path = os.path.join(self.config.output_dir, "naive_vs_replay_comparison.png")
            if os.path.exists(naive_csv):
                plot_naive_vs_replay_comparison(naive_csv_path=naive_csv, replay_tracker=self.tracker, save_path=comp_path)

        print(f"[Visualizations] All plots successfully saved in: {self.config.output_dir}")

        return self.tracker
