"""Configuration dataclasses and hyperparameters for continual learning experiments."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Experiment Hyperparameters & Runtime Configuration."""
    data_dir: str = "./data"
    output_dir: str = "./results"
    checkpoints_dir: str = "./checkpoints"
    
    # Model & Backbone
    pretrained_backbone: bool = True
    dropout_rate: float = 0.5
    
    # Continual Learning Setup
    num_phases: int = 4
    superclasses_per_phase: int = 5
    classes_per_phase: int = 25  # 5 superclasses * 5 fine classes
    
    # Training
    epochs_per_phase: int = 15
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    eval_every_epoch: bool = False
    
    # Isolated Baselines
    run_isolated_baselines: bool = True
    baseline_epochs: int = 15
    force_baselines: bool = False

    # Experience Replay Buffer
    use_replay: bool = False
    replay_samples_per_class: int = 20

    # Hardware & Seed
    device: Optional[str] = None
    seed: int = 42
