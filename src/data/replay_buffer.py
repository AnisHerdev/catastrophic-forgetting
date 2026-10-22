"""
Replay Buffer (Exemplar Memory) for Class-Incremental Continual Learning.
Stores a small, fixed subset of training samples from completed phases
and reconstructs training streams to mitigate catastrophic forgetting.
"""

import random
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import torchvision


class ReplayBuffer(Dataset):
    """
    Exemplar Memory Dataset.
    Maintains indices of training samples from previously learned classes
    and loads them dynamically with transforms.
    """

    def __init__(
        self,
        base_dataset: torchvision.datasets.CIFAR100,
        class_mapping: Dict[int, int],
        samples_per_class: int = 20,
    ):
        """
        Args:
            base_dataset: Raw training dataset (with training augmentations).
            class_mapping: Global class ID mapping (raw_class_id -> contiguous_id).
            samples_per_class: Number of exemplars to store per class.
        """
        self.base_dataset = base_dataset
        self.class_mapping = class_mapping
        self.samples_per_class = samples_per_class
        self.stored_indices: List[int] = []
        self.seen_classes: List[int] = []

    def __len__(self) -> int:
        return len(self.stored_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.stored_indices[idx]
        image, raw_target = self.base_dataset[real_idx]
        remapped_target = self.class_mapping[raw_target]
        return image, remapped_target

    def add_phase_classes(
        self,
        phase_classes: List[int],
        seed: int = 42,
    ) -> int:
        """
        Deterministically samples `samples_per_class` exemplars for each class in `phase_classes`
        and adds their dataset indices to the replay buffer.

        Args:
            phase_classes: List of raw fine class IDs trained in the completed phase.
            seed: Base RNG seed for reproducible exemplar selection.

        Returns:
            Number of new exemplars added.
        """
        initial_count = len(self.stored_indices)
        targets = self.base_dataset.targets

        for raw_c in phase_classes:
            if raw_c in self.seen_classes:
                continue

            # Find all candidate indices for this class in raw training data
            class_indices = [i for i, t in enumerate(targets) if t == raw_c]

            # Deterministic pseudo-random selection per class
            rng = random.Random(seed + raw_c)
            num_to_sample = min(self.samples_per_class, len(class_indices))
            selected = rng.sample(class_indices, num_to_sample)

            self.stored_indices.extend(selected)
            self.seen_classes.append(raw_c)

        added = len(self.stored_indices) - initial_count
        return added

    def clear(self) -> None:
        """Clears all stored exemplars."""
        self.stored_indices.clear()
        self.seen_classes.clear()
