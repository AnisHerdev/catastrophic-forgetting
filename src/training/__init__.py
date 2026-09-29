"""Training pipeline for continual learning."""
from src.training.trainer import ContinualLearningPipeline, train_model, train_single_epoch

__all__ = ["ContinualLearningPipeline", "train_model", "train_single_epoch"]
