"""
Evaluation and Catastrophic Forgetting Metrics Engine.
Computes task-wise accuracy matrices, catastrophic forgetting scores,
backward transfer (BWT), and average accuracy across continual learning phases.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates classification accuracy and average loss on a DataLoader.

    Returns:
        (accuracy_percentage, average_loss)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_samples = 0
    correct_predictions = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    accuracy = (correct_predictions / total_samples) * 100.0 if total_samples > 0 else 0.0

    return accuracy, avg_loss


class ContinualMetricsTracker:
    """
    Tracks and computes the 4x4 evaluation matrix R_{i,j},
    catastrophic forgetting drops, average accuracies, and baseline comparisons.
    """

    def __init__(self, num_phases: int = 4):
        self.num_phases = num_phases
        # R[i, j] stores accuracy on Phase (j+1) test set after completing Phase (i+1)
        self.accuracy_matrix = np.full((num_phases, num_phases), np.nan)
        # Stores isolated single-phase baseline accuracies
        self.baseline_accuracies: Dict[int, float] = {}
        # Stores cumulative test set accuracy after each phase
        self.cumulative_accuracies: Dict[int, float] = {}

    def record_accuracy(self, trained_phase: int, evaluated_phase: int, accuracy: float) -> None:
        """
        Record accuracy on evaluated_phase test set after trained_phase training.
        Phases are 1-indexed.
        """
        self.accuracy_matrix[trained_phase - 1, evaluated_phase - 1] = accuracy

    def record_baseline(self, phase: int, accuracy: float) -> None:
        """Record isolated single-phase baseline accuracy for phase (1-indexed)."""
        self.baseline_accuracies[phase] = accuracy

    def record_cumulative(self, phase: int, accuracy: float) -> None:
        """Record accuracy on cumulative (all seen so far) test set after phase (1-indexed)."""
        self.cumulative_accuracies[phase] = accuracy

    def get_forgetting_per_phase(self) -> Dict[int, float]:
        """
        Computes catastrophic forgetting for each phase j < num_phases:
        F_j = R_{j, j} - R_{num_phases, j}
        """
        forgetting = {}
        final_phase_idx = self.num_phases - 1
        for j in range(self.num_phases - 1):
            initial_acc = self.accuracy_matrix[j, j]
            final_acc = self.accuracy_matrix[final_phase_idx, j]
            if not np.isnan(initial_acc) and not np.isnan(final_acc):
                forgetting[j + 1] = initial_acc - final_acc
        return forgetting

    def get_average_forgetting(self) -> float:
        """Computes average catastrophic forgetting across all completed prior phases."""
        drops = self.get_forgetting_per_phase()
        return float(np.mean(list(drops.values()))) if drops else 0.0

    def get_backward_transfer(self) -> float:
        """
        Backward Transfer (BWT) = 1 / (N - 1) * sum_{j=1}^{N-1} (R_{N, j} - R_{j, j})
        Negative BWT indicates catastrophic forgetting.
        """
        return -self.get_average_forgetting()

    def get_average_accuracy_after_phase(self, phase: int) -> float:
        """Average accuracy across all tasks seen up to `phase` evaluated at `phase`."""
        idx = phase - 1
        valid_accs = [self.accuracy_matrix[idx, j] for j in range(phase) if not np.isnan(self.accuracy_matrix[idx, j])]
        return float(np.mean(valid_accs)) if valid_accs else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Exports the R_{i,j} accuracy matrix as a structured pandas DataFrame."""
        columns = [f"Eval Phase {j+1}" for j in range(self.num_phases)]
        index = [f"After Training Phase {i+1}" for i in range(self.num_phases)]
        df = pd.DataFrame(self.accuracy_matrix, index=index, columns=columns)
        
        # Add summary columns
        avg_accs = [self.get_average_accuracy_after_phase(i + 1) for i in range(self.num_phases)]
        df["Average Accuracy (Seen)"] = avg_accs
        if self.cumulative_accuracies:
            df["Cumulative Test Acc"] = [self.cumulative_accuracies.get(i + 1, np.nan) for i in range(self.num_phases)]
        return df

    def save_csv(self, filepath: str) -> None:
        """Saves detailed metrics to a CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=True)

    def print_summary_table(self) -> None:
        """Prints a comprehensive tabular report in terminal."""
        print("\n" + "=" * 80)
        print("CONTINUAL LEARNING ACCURACY MATRIX R_{i,j} (%):")
        print("=" * 80)
        header = f"{'Training Phase':<25}" + "".join([f"{'Eval P' + str(j+1):<12}" for j in range(self.num_phases)]) + f"{'Avg (Seen)':<12}"
        print(header)
        print("-" * 80)

        for i in range(self.num_phases):
            row_str = f"After Phase {i+1} Training    "
            for j in range(self.num_phases):
                val = self.accuracy_matrix[i, j]
                row_str += f"{val:10.2f}%  " if not np.isnan(val) else f"{'-':^10}  "
            avg_val = self.get_average_accuracy_after_phase(i + 1)
            row_str += f"{avg_val:10.2f}%"
            print(row_str)

        print("=" * 80)
        print("CATASTROPHIC FORGETTING ANALYSIS:")
        print("-" * 80)
        forgetting = self.get_forgetting_per_phase()
        for p, drop in forgetting.items():
            initial = self.accuracy_matrix[p - 1, p - 1]
            final = self.accuracy_matrix[self.num_phases - 1, p - 1]
            baseline_str = f" | Isolated Baseline: {self.baseline_accuracies[p]:.2f}%" if p in self.baseline_accuracies else ""
            print(f"Phase {p} Accuracy: Initial = {initial:.2f}%  -->  Final = {final:.2f}%  |  Forgetting Drop = {drop:+.2f}%{baseline_str}")

        print("-" * 80)
        print(f"Average Catastrophic Forgetting : {self.get_average_forgetting():.2f}%")
        print(f"Backward Transfer (BWT)          : {self.get_backward_transfer():+.2f}%")
        print(f"Final Model Average Accuracy     : {self.get_average_accuracy_after_phase(self.num_phases):.2f}%")
        print("=" * 80 + "\n")


@dataclass
class EpochRecord:
    """Record of a single phase evaluation at a specific training epoch."""
    training_phase: int
    epoch: int
    global_epoch: int
    evaluated_phase: int
    accuracy: float
    loss: float
    initial_peak_accuracy: Optional[float] = None
    forgetting_drop: Optional[float] = None


class EpochForgettingTracker:
    """
    Tracks epoch-by-epoch evaluation across all continual learning phases
    to quantify real-time catastrophic forgetting and retention trajectories.
    """

    def __init__(self):
        self.records: List[EpochRecord] = []

    def record(
        self,
        training_phase: int,
        epoch: int,
        global_epoch: int,
        evaluated_phase: int,
        accuracy: float,
        loss: float,
        initial_peak_accuracy: Optional[float] = None,
    ) -> None:
        """
        Record performance on an evaluated phase test set at a specific epoch of training.
        """
        forgetting_drop = None
        if initial_peak_accuracy is not None and evaluated_phase < training_phase:
            forgetting_drop = initial_peak_accuracy - accuracy

        record = EpochRecord(
            training_phase=training_phase,
            epoch=epoch,
            global_epoch=global_epoch,
            evaluated_phase=evaluated_phase,
            accuracy=accuracy,
            loss=loss,
            initial_peak_accuracy=initial_peak_accuracy,
            forgetting_drop=forgetting_drop,
        )
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Exports the epoch records to a structured pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "training_phase",
                    "epoch",
                    "global_epoch",
                    "evaluated_phase",
                    "accuracy",
                    "loss",
                    "initial_peak_accuracy",
                    "forgetting_drop",
                ]
            )
        return pd.DataFrame([asdict(r) for r in self.records])

    def save_csv(self, filepath: str) -> None:
        """Saves epoch tracking metrics to CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def find_collapse_epochs(self, chance_threshold: float = 5.0) -> Dict[Tuple[int, int], Optional[int]]:
        """
        Identifies the first epoch in a subsequent training phase where
        the previous phase's accuracy collapses below chance_threshold (e.g. 5.0%).
        Returns mapping: (training_phase, evaluated_phase) -> first collapse epoch or None.
        """
        df = self.to_dataframe()
        collapse_map: Dict[Tuple[int, int], Optional[int]] = {}

        if df.empty:
            return collapse_map

        for (tp, ep), group in df.groupby(["training_phase", "evaluated_phase"]):
            if ep < tp:  # Only for prior phases evaluated during subsequent training
                collapsed = group[group["accuracy"] <= chance_threshold]
                if not collapsed.empty:
                    first_epoch = int(collapsed["epoch"].iloc[0])
                    collapse_map[(int(tp), int(ep))] = first_epoch
                else:
                    collapse_map[(int(tp), int(ep))] = None

        return collapse_map

