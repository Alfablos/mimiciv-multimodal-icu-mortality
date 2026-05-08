# In-Hospital Multimodal ICU Mortality Prediction

This project estimates in-hospital death risk for patients just admitted to the ICU. It uses data from the 24 hours before ICU admission up to ICU admission time, where `T=0` is `mimiciv_icu.icustays.intime`.

The model is intended for early ICU admission prioritization. It combines chest X-ray data with tabular clinical features from vitals and labs.

General rules:

- Only the first ICU admission per patient is used, to avoid temporal leakage.
- If multiple X-rays are available, the image closest to ICU admission is selected.
- X-ray timing uses `StudyDate` and `StudyTime`, not report timestamps.
- Lab events and vital signs are filtered by `charttime`.
- Admission diagnosis is not used because it may be added later and can contain human error.

Datasets:

- MIMIC-IV
- MIMIC-ED
- MIMIC-CXR/MIMIC-CXR-JPG

## Architecture

The model is a composite neural network with:

- A *visual encoder* for chest X-ray image features.
- A *tabular encoder* for vitals, labs, demographics, and missingness indicators.
- A *fusion* model that combines both representations to predict in-hospital mortality.

## Project Structure

This repository contains two independent Python projects inside a single repo:

- `generator/` — Dataset generation pipeline (reads MIMIC databases, produces the artifact bundle)
- `trainer/` — Model training pipeline (consumes artifact bundle, trains the multimodal model)

Each project has its own `pyproject.toml`, dependencies, Dockerfile, Compose file, and test suite. The only runtime contract between them is the dataset artifact bundle described below.

A shared root `justfile` delegates commands to each project. Run `just --list` to see all available recipes.

## Quick Start

### Generator — Build a Dataset

```bash
# Run locally (requires DuckDB with MIMIC-IV + MIMIC-ED tables)
just generator build-dataset

# Or pass custom args
just generator -- --help

# Build and run via Docker Compose
just generator-compose-up
```

### Trainer — Train the Model

```bash
# Run locally (requires dataset/ directory with artifacts)
just trainer train

# Or via Docker Compose (mounts dataset/ into /app/dataset)
just trainer-compose-up
```

### Run All Tests

```bash
just test                  # All tests (root aggregate)
just generator-test        # Generator-only tests
just trainer-test          # Trainer-only tests
just check-boundaries      # Prove generator/trainer split is intact
```

## Generator

The generator reads MIMIC-IV and MIMIC-ED clinical data through a DuckDB database, links it with MIMIC-CXR metadata, and produces a dataset artifact bundle consumed by the trainer.

### Local Workflow

```bash
# Install generator dependencies
uv sync --project generator

# Show available commands
uv --project generator run python -m generator.main --help

# Build a complete dataset
uv --project generator run python -m generator.main build-dataset
```

### Container Workflow

The generator Compose file defines a `generator` service with safe defaults. It does not automatically generate data — operators add volume mounts and override the command intentionally.

```bash
just generator-compose-up
```

Generator-specific environment variables for optional LakeFS integration are available in `generator/compose.yml`.

## Trainer

The trainer consumes the dataset artifact bundle and trains a multimodal model combining chest X-ray image features with tabular clinical features.

### Local Workflow

```bash
# Install trainer dependencies
uv sync --project trainer

# Show available commands
uv --project trainer run python -m trainer.main --help

# Train with default config
uv --project trainer run python -m trainer.main train
```

### Container Workflow

The trainer Compose file mounts a local `dataset/` directory into `/app/dataset` and starts training via `trainer/entrypoint.sh` which validates the full dataset bundle before launching.

```bash
just trainer-compose-up
```

The trainer entrypoint (`trainer/entrypoint.sh`) checks for all 6 expected files + the image tree directory before starting. If any are missing, it prints the full list and exits with code 1.

## Dataset Artifact Contract

The generator produces a complete artifact bundle. The trainer consumes it. This is their only runtime contract.

### Full Generator Output Bundle

| Artifact | Produced by | Role |
|----------|-------------|------|
| `ds_train.csv` | `generator/builder.py` | Training rows consumed through `TRAINING_DATASET_FILE` |
| `ds_val.csv` | `generator/builder.py` | Validation rows consumed through `VALIDATION_DATASET_FILE` |
| `ds_test.csv` | `generator/builder.py` | Held-out test split (part of the complete bundle) |
| `stats.json` | `generator/builder.py` | Training-set means and standard deviations |
| `manifest.json` | `generator/builder.py` | Provenance and bundle integrity |
| `schema.json` | `generator/builder.py` | Column and role metadata |
| Image tree | Operator populates from MIMIC-CXR-JPG | Files resolved under `DATASET_IMAGES_BASEDIR` |

`generator/builder.py` is the canonical producer. The trainer consumes artifacts — it must not define an alternate contract, invoke builder logic, or select storage keys.

### Runtime Path Conventions

**Container:** The trainer working directory is `/app`. The dataset directory is `/app/dataset`.

**Local:** The equivalent local convention is `./dataset/`.

```
/app/dataset/
├── ds_train.csv          # Training split
├── ds_val.csv            # Validation split
├── ds_test.csv           # Test split
├── stats.json            # Dataset statistics
├── manifest.json         # Provenance metadata
├── schema.json           # Column metadata
└── images/               # Image tree (mounted or populated by operator)
    └── mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.1.0/files/
        └── pXX/pXXXXXXXX/sXXXXXXXX/<dicom_id>.jpg
```

The operator decides how to populate or mount the dataset directory. The trainer's runtime contract is the filesystem shape at startup, not the storage mechanism.

### Fail-fast and scope boundaries

Missing required artifacts are configuration or data errors. They fail fast with clear messages and must not trigger `trainer/builder.py` fallback behavior, S3 key selection, or per-file download choices. `trainer/builder.py` must not be treated as a second contract source.

This contract keeps `generator/` and `trainer/` coupled by files only, not by runtime Python imports.

## Trainer Configuration

The trainer reads configuration from environment variables.

### Dataset Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING_DATASET_FILE` | `./dataset/ds_train.csv` | Training CSV path |
| `VALIDATION_DATASET_FILE` | `./dataset/ds_val.csv` | Validation CSV path |
| `DATASET_STATS_FILE` | `./dataset/stats.json` | Dataset statistics JSON |
| `DATASET_IMAGES_BASEDIR` | `./dataset/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.1.0/files` | Image tree base directory |
| `DATASET_IMAGES_EXTENSION` | `jpg` | Image file extension (jpg or jpeg) |

### MLflow

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | Set via Compose | MLflow tracking server URI |
| `MLFLOW_TRACKING_USERNAME` | Set via Compose | MLflow auth username |
| `MLFLOW_TRACKING_PASSWORD` | Set via Compose | MLflow auth password |
| `MLFLOW_EXPERIMENT_NAME` | `Multimodal ICU mortality` | Experiment name |

### Training Hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MMIM_BATCH_SIZE` | `32` | Batch size |
| `MMIM_EPOCHS` | `10` | Number of training epochs |
| `MMIM_DROPOUT` | `0.3` | Dropout rate for fusion model |
| `MMIM_LEARNING_RATE` | `0.001` | AdamW learning rate |
| `MMIM_TRAIN_LIMIT` | `1.0` | Fraction of data to use (for faster iteration) |
| `MMIM_NUM_WORKERS` | Auto (CPU-based) | DataLoader workers |
| `MMIM_DEBUG` | `false` | Enable dataset debug output |
| `MMIM_DATASET_SHUFFLE` | `true` | Enable DataLoader shuffle |
| `MMIM_LOSS_POS_WEIGHT` | `5160 / 936` | Pos class weight for imbalanced labels |

## Limitations

- Data is sparse because the project uses ICU stays while looking back up to 24 hours before ICU admission.
- The model currently trains on one image per selected stay, so it does not model time trends across multiple images.
- Tabular time-series data is flattened into values such as `_min`, `_max`, and `_mean`.
