"""Data loading and partition modules for CIFAR-100 continual learning."""
from src.data.cifar100_split import (
    CIFAR100ContinualDataModule,
    SUPERCLASS_NAMES,
    SUPERCLASS_TO_FINE_CLASSES,
)

__all__ = [
    "CIFAR100ContinualDataModule",
    "SUPERCLASS_NAMES",
    "SUPERCLASS_TO_FINE_CLASSES",
]
