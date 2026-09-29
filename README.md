# Quantifying Catastrophic Forgetting in Class-Incremental Learning with VGG16 on CIFAR-100

This repository provides a modular, reproducible implementation for studying and quantifying **Catastrophic Forgetting (CF)** in deep neural networks within a **Class-Incremental Continual Learning** setting on the **CIFAR-100** dataset using an adapted **VGG16** architecture.

---

## 🔬 Experimental Methodology

### 1. Sequential Phase Partitioning (CIFAR-100)
CIFAR-100 consists of 100 fine classes grouped into 20 superclasses (5 fine classes per superclass). The classes are partitioned into **4 disjoint sequential phases** (5 superclasses = 25 fine classes per phase):

| Phase | Superclasses | Coarse Superclass Names | Fine Class Range | Total Seen |
|---|---|---|---|---|
| **Phase 1** | 0 – 4 | Aquatic mammals, Fish, Flowers, Food containers, Fruit & veg | 25 classes (`0..24`) | 25 |
| **Phase 2** | 5 – 9 | Electrical devices, Furniture, Insects, Large carnivores, Outdoor things | 25 classes (`25..49`) | 50 |
| **Phase 3** | 10 – 14 | Natural scenes, Omnivores/herbivores, Medium mammals, Invertebrates, People | 25 classes (`50..74`) | 75 |
| **Phase 4** | 15 – 19 | Reptiles, Small mammals, Trees, Vehicles 1, Vehicles 2 | 25 classes (`75..99`) | 100 |

### 2. VGG16-Adapted Architecture
- **Backbone**: Pre-trained ImageNet `VGG16` convolutional layers (`vgg16(weights=IMAGENET1K_V1)`).
- **Pooling**: Adaptive Average Pooling `AdaptiveAvgPool2d((1, 1))` yielding 512-dimensional feature vectors from $32 \times 32$ CIFAR-100 images.
- **Classification Head**:
  $$\text{Linear}(512 \rightarrow 256) \longrightarrow \text{ReLU} \longrightarrow \text{Dropout}(p=0.5) \longrightarrow \text{Linear}(256 \rightarrow N_{\text{classes}})$$
- **Dynamic Head Expansion**:
  When transitioning from Phase $k-1$ to Phase $k$, the output dimension expands dynamically from $C_{\text{old}}$ to $C_{\text{new}}$:
  - Previously learned weights and biases for classes $[0 \dots C_{\text{old}}-1]$ are strictly preserved.
  - New output weights $[C_{\text{old}} \dots C_{\text{new}}-1]$ are initialized using Kaiming normal.
  - The model is fine-tuned end-to-end on the new phase data.

### 3. Quantification Metrics
- **Accuracy Matrix ($R_{i,j}$)**: Accuracy on Phase $j$ test set after completing training on Phase $i$.
- **Catastrophic Forgetting ($F_j$)**: Drop in accuracy on Phase $j$ between initial peak training ($R_{j,j}$) and the final state ($R_{4,j}$):
  $$F_j = R_{j,j} - R_{4,j}$$
- **Average Forgetting**: $\frac{1}{3} \sum_{j=1}^{3} F_j$
- **Backward Transfer (BWT)**: $\text{BWT} = \frac{1}{N-1}\sum_{j=1}^{N-1}(R_{N,j} - R_{j,j}) = -\text{Avg}(F)$
- **Isolated Single-Phase Baselines ($B_j$)**: Independent models trained exclusively on each phase's 25 classes to establish the reference upper-bound performance.

---

## 📁 Codebase Layout

```
catastrophic-forgetting/
├── pyproject.toml               # uv project dependencies
├── README.md                    # Project overview and CLI guide
├── main.py                      # CLI entrypoint with configurable hyperparams
├── src/
│   ├── data/
│   │   └── cifar100_split.py   # Superclass partitioning & DataLoaders
│   ├── models/
│   │   └── vgg16_adapted.py    # VGG16 with dynamically expandable head
│   ├── training/
│   │   └── trainer.py          # Continual learning pipeline & baseline trainer
│   ├── evaluation/
│   │   └── metrics.py          # Accuracy matrix, forgetting & BWT calculations
│   └── utils/
│       ├── config.py           # Experiment configuration dataclass
│       └── visualizer.py       # Heatmaps, accuracy decay curves, bar charts
└── results/                     # Generated CSV metrics & visualization plots
```

---

## ⚡ Usage & Execution

### 1. Environment Setup with `uv`

```powershell
# Activate the virtual environment
.venv\Scripts\activate

# Install dependencies (already completed)
uv add numpy pandas matplotlib seaborn tqdm scikit-learn
```

### 2. Fast Sanity Check (1 Epoch per Phase)
To verify the full pipeline and visualization generation quickly:
```powershell
uv run python main.py --quick-test
```

### 3. Full Continual Learning Experiment
To run the full 4-phase continual learning experiment (with isolated baselines):
```powershell
uv run python main.py --epochs 15 --batch-size 64 --lr 0.001
```

### 4. Continual Learning Only (Skip Baselines)
```powershell
uv run python main.py --epochs 15 --no-baselines
```

### 5. CLI Arguments Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--epochs` | `int` | `15` | Training epochs per continual phase |
| `--batch-size` | `int` | `64` | Mini-batch size |
| `--lr` | `float` | `0.001` | Initial Adam learning rate |
| `--weight-decay`| `float` | `0.0001`| Weight decay regularization |
| `--dropout` | `float` | `0.5` | Dropout probability in classifier head |
| `--no-baselines`| flag | `False` | Skip running isolated single-phase baselines |
| `--baseline-epochs` | `int` | `15` | Training epochs for isolated baseline models |
| `--data-dir` | `str` | `./data` | Directory where CIFAR-100 dataset is stored |
| `--output-dir` | `str` | `./results` | Directory for output CSV files and PNG plots |
| `--checkpoints-dir` | `str` | `./checkpoints` | Directory to save `.pt` model weights |
| `--quick-test` | flag | `False` | Run a 1-epoch smoke test across all 4 phases |

---

## 📊 Generated Outputs

All outputs are saved to `./results/`:
1. **`continual_forgetting_results.csv`**: Full $4 \times 4$ accuracy matrix $R_{i,j}$, average accuracy, and cumulative test accuracy.
2. **`accuracy_matrix_heatmap.png`**: Heatmap showing retention across phases and forgetting severity.
3. **`accuracy_decay_curves.png`**: Line plot tracking the performance decline of Phase 1, 2, and 3 as subsequent phases are learned.
4. **`continual_vs_baseline.png`**: Bar chart comparing Initial Accuracy vs Retained Accuracy vs Isolated Baseline for each phase.
