"""
CIFAR-100 Superclass Partitioning and Continual Learning DataLoaders.
Partitions the 20 superclasses of CIFAR-100 into 4 sequential phases (5 superclasses / 25 fine classes per phase).
"""

from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms


# CIFAR-100 Superclasses to Fine-Classes mapping (Standard CIFAR-100 definition)
SUPERCLASS_TO_FINE_CLASSES: Dict[int, List[int]] = {
    0: [4, 30, 55, 72, 95],       # aquatic mammals: beaver, dolphin, otter, seal, whale
    1: [1, 32, 67, 73, 91],       # fish: aquarium fish, flatfish, ray, shark, trout
    2: [54, 62, 70, 82, 92],      # flowers: orchids, poppies, roses, sunflowers, tulips
    3: [9, 10, 16, 28, 61],       # food containers: bottles, bowls, cans, cups, plates
    4: [0, 51, 53, 57, 83],       # fruit and vegetables: apples, mushrooms, oranges, pears, sweet peppers
    5: [22, 39, 40, 86, 87],      # household electrical devices: clock, computer keyboard, lamp, telephone, television
    6: [5, 20, 25, 84, 94],       # household furniture: bed, chair, couch, table, wardrobe
    7: [6, 7, 14, 18, 24],        # insects: bee, beetle, butterfly, caterpillar, cockroach
    8: [3, 42, 43, 88, 97],       # large carnivores: bear, leopard, lion, tiger, wolf
    9: [12, 17, 37, 68, 76],      # large man-made outdoor things: bridge, castle, house, road, skyscraper
    10: [23, 33, 49, 60, 71],     # large natural outdoor scenes: cloud, forest, mountain, plain, sea
    11: [15, 19, 21, 31, 38],     # large omnivores and herbivores: camel, cattle, chimpanzee, elephant, kangaroo
    12: [34, 63, 64, 66, 75],     # medium-sized mammals: fox, porcupine, possum, raccoon, skunk
    13: [26, 45, 77, 79, 99],     # non-insect invertebrates: crab, lobster, snail, spider, worm
    14: [2, 11, 35, 46, 98],      # people: baby, boy, girl, man, woman
    15: [27, 29, 44, 78, 93],     # reptiles: crocodile, dinosaur, lizard, snake, turtle
    16: [36, 50, 65, 74, 80],     # small mammals: hamster, mouse, rabbit, shrew, squirrel
    17: [47, 52, 56, 59, 96],     # trees: maple, oak, palm, pine, willow
    18: [8, 13, 48, 58, 90],      # vehicles 1: bicycle, bus, motorcycle, pickup truck, train
    19: [41, 69, 81, 85, 89],     # vehicles 2: lawn-mower, rocket, streetcar, tank, tractor
}

SUPERCLASS_NAMES: Dict[int, str] = {
    0: "aquatic mammals",
    1: "fish",
    2: "flowers",
    3: "food containers",
    4: "fruit and vegetables",
    5: "household electrical devices",
    6: "household furniture",
    7: "insects",
    8: "large carnivores",
    9: "large man-made outdoor things",
    10: "large natural outdoor scenes",
    11: "large omnivores and herbivores",
    12: "medium-sized mammals",
    13: "non-insect invertebrates",
    14: "people",
    15: "reptiles",
    16: "small mammals",
    17: "trees",
    18: "vehicles 1",
    19: "vehicles 2",
}

# ImageNet normalization parameters (standard for pre-trained VGG16)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PhaseSubsetDataset(Dataset):
    """
    Wraps a CIFAR-100 dataset subset and remaps the original fine class IDs
    to target class indices for continual learning or isolated baselines.
    """
    def __init__(
        self,
        base_dataset: torchvision.datasets.CIFAR100,
        indices: List[int],
        class_mapping: Dict[int, int],
    ):
        self.base_dataset = base_dataset
        self.indices = indices
        self.class_mapping = class_mapping

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        image, raw_target = self.base_dataset[real_idx]
        remapped_target = self.class_mapping[raw_target]
        return image, remapped_target


class CIFAR100ContinualDataModule:
    """
    Manages CIFAR-100 downloads, superclass partitioning, and DataLoader creation
    for 4-phase continual class-incremental learning and isolated single-phase baselines.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 2,
        superclasses_per_phase: int = 5,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.superclasses_per_phase = superclasses_per_phase
        self.num_phases = 20 // superclasses_per_phase  # 4 phases by default

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Download raw datasets
        self.raw_train_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=self.train_transform
        )
        self.raw_test_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=self.test_transform
        )

        # Build phase partitions
        self.phase_fine_classes: Dict[int, List[int]] = {}
        self.phase_superclass_ids: Dict[int, List[int]] = {}
        self.global_class_mapping: Dict[int, int] = {}
        self.phase_local_mapping: Dict[int, Dict[int, int]] = {}

        self._setup_partitions()

    def _setup_partitions(self) -> None:
        """Partition 20 superclasses across phases and build class mappings."""
        global_id_counter = 0

        for phase in range(1, self.num_phases + 1):
            sc_start = (phase - 1) * self.superclasses_per_phase
            sc_end = phase * self.superclasses_per_phase
            superclasses = list(range(sc_start, sc_end))
            self.phase_superclass_ids[phase] = superclasses

            fine_classes: List[int] = []
            phase_local_map: Dict[int, int] = {}
            local_id = 0

            for sc_id in superclasses:
                classes = SUPERCLASS_TO_FINE_CLASSES[sc_id]
                for raw_c in classes:
                    fine_classes.append(raw_c)
                    self.global_class_mapping[raw_c] = global_id_counter
                    phase_local_map[raw_c] = local_id
                    global_id_counter += 1
                    local_id += 1

            self.phase_fine_classes[phase] = fine_classes
            self.phase_local_mapping[phase] = phase_local_map

    def _get_subset_indices(self, raw_dataset: torchvision.datasets.CIFAR100, target_classes: List[int]) -> List[int]:
        target_set = set(target_classes)
        return [i for i, label in enumerate(raw_dataset.targets) if label in target_set]

    def get_phase_dataloaders(
        self, phase: int, mode: str = "continual"
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Returns (train_loader, test_loader) for a specific phase.
        
        Args:
            phase: Phase index (1 to num_phases).
            mode: "continual" uses globally contiguous labels (0..24 for phase 1, 25..49 for phase 2, etc.),
                  "isolated" uses local labels (0..24) for single-phase baseline training.
        """
        if phase not in self.phase_fine_classes:
            raise ValueError(f"Invalid phase {phase}. Must be between 1 and {self.num_phases}.")

        classes = self.phase_fine_classes[phase]
        mapping = self.global_class_mapping if mode == "continual" else self.phase_local_mapping[phase]

        train_indices = self._get_subset_indices(self.raw_train_dataset, classes)
        test_indices = self._get_subset_indices(self.raw_test_dataset, classes)

        train_subset = PhaseSubsetDataset(self.raw_train_dataset, train_indices, mapping)
        test_subset = PhaseSubsetDataset(self.raw_test_dataset, test_indices, mapping)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, test_loader

    def get_cumulative_test_loader(self, up_to_phase: int) -> DataLoader:
        """Returns a test loader containing all classes seen from Phase 1 up to `up_to_phase`."""
        all_classes = []
        for p in range(1, up_to_phase + 1):
            all_classes.extend(self.phase_fine_classes[p])

        test_indices = self._get_subset_indices(self.raw_test_dataset, all_classes)
        test_subset = PhaseSubsetDataset(self.raw_test_dataset, test_indices, self.global_class_mapping)

        return DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def print_phase_summary(self) -> None:
        """Prints a readable summary of superclasses and class counts per phase."""
        print("=" * 70)
        print("CIFAR-100 CONTINUAL LEARNING PARTITIONS SUMMARY")
        print("=" * 70)
        for phase, superclasses in self.phase_superclass_ids.items():
            sc_names = [SUPERCLASS_NAMES[sc] for sc in superclasses]
            fine_cls = self.phase_fine_classes[phase]
            print(f"Phase {phase}:")
            print(f"  - Superclasses ({len(superclasses)}): {', '.join(sc_names)}")
            print(f"  - Number of Fine Classes: {len(fine_cls)} (Global IDs: {min(self.global_class_mapping[c] for c in fine_cls)} to {max(self.global_class_mapping[c] for c in fine_cls)})")
        print("=" * 70)
