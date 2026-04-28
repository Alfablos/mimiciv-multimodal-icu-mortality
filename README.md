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

## Configuration

The project is configured through environment variables. They are used in three different phases:

- Docker image build time: `Dockerfile` accepts source metadata as build args and stores it in image labels and environment variables.
- Container startup time: `entrypoint.sh` resolves dataset paths, downloads missing files from S3-compatible storage, then launches the Python CLI entrypoint.
- Python runtime: `trainer/config.py`, `trainer/train.py`, `trainer/meta.py`, and `trainer/builder.py` read variables for training, metadata logging, and dataset generation.

### Build Metadata

These variables identify the source revision used for a training run.

| Variable | Default | Used by | Description |
| --- | --- | --- | --- |
| `IMAGE_NAME` | None | `justfile` | Docker image name used by `just image`. Required for that command. |
| `GIT_SHA` | Set by `just image` from `git rev-parse HEAD` | `Dockerfile`, `trainer/meta.py` | Source commit recorded in image labels and MLflow metadata. |
| `GIT_REF` | Set by `just image` from the current branch | `Dockerfile`, `trainer/meta.py` | Source branch or ref recorded in image labels and MLflow metadata. |

Build an image locally:

```bash
IMAGE_NAME=registry.example.com/mmim/mmim-trainer just image
```

Equivalent manual metadata commands:

```bash
export GIT_SHA="$(git rev-parse HEAD)"
export GIT_REF="$(git symbolic-ref --quiet --short HEAD || git rev-parse --short HEAD)"
```

The `Dockerfile` consumes these as build args:

```bash
docker build \
  --build-arg GIT_SHA="$GIT_SHA" \
  --build-arg GIT_REF="$GIT_REF" \
  -t "$IMAGE_NAME:$GIT_REF-$GIT_SHA" .
```

Note: `docker compose up --build` does not currently pass `GIT_SHA` or `GIT_REF` as build args unless `build.args` is added to `docker-compose.yml`.

### Dataset Paths

These variables point the trainer and dataset builder to local files. In Docker, `entrypoint.sh` derives defaults from `BASE_DIR` and `DATASET_LOCAL_DIR`.

| Variable | Default | Used by | Description |
| --- | --- | --- | --- |
| `BASE_DIR` | `/app` in Docker | `entrypoint.sh` | Directory the entrypoint enters before launching Python. |
| `DATASET_LOCAL_DIR` | `$BASE_DIR/dataset` in Docker | `entrypoint.sh` | Base local directory for dataset files and images. |
| `TRAINING_DATASET_FILE` | `$DATASET_LOCAL_DIR/ds_train.csv` in Docker, `./dataset/ds_train.csv` locally | `entrypoint.sh`, `trainer/config.py`, `trainer/builder.py`, `trainer/meta.py` | Training CSV path. |
| `VALIDATION_DATASET_FILE` | `$DATASET_LOCAL_DIR/ds_val.csv` in Docker, `./dataset/ds_val.csv` locally | `entrypoint.sh`, `trainer/config.py`, `trainer/builder.py`, `trainer/meta.py` | Validation CSV path. |
| `TEST_DATASET_FILE` | `./ds_test.csv` in dataset builder | `trainer/builder.py` | Test CSV output path when building datasets. |
| `DATASET_STATS_FILE` | `$DATASET_LOCAL_DIR/stats.json` in Docker, `./dataset/stats.json` locally | `entrypoint.sh`, `trainer/config.py`, `trainer/builder.py`, `trainer/meta.py` | JSON file containing training-set means and standard deviations. |
| `DATASET_IMAGES_BASEDIR` | `$DATASET_LOCAL_DIR/$DATASET_IMAGES_DIR` in Docker, `./dataset/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.1.0/files` locally | `entrypoint.sh`, `trainer/config.py`, `trainer/meta.py` | Local base directory containing MIMIC-style image paths. |
| `DATASET_IMAGES_EXTENSION` | `dcm` in `entrypoint.sh` and `trainer/config.py`, `jpg` in `docker-compose.yml` | `entrypoint.sh`, `trainer/config.py`, `trainer/meta.py` | Image extension passed to the dataset loader. Use `jpg`, `.jpg`, `dcm`, `.dcm`, `dicom`, or `.dicom`. |

`docker-compose.yml` mounts the local dataset into the container:

```yaml
volumes:
  - ./dataset/ds_train.csv:/app/dataset/ds_train.csv
  - ./dataset/ds_val.csv:/app/dataset/ds_val.csv
  - ./dataset/stats.json:/app/dataset/stats.json
  - ./dataset/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.1.0/files:/app/dataset/mimic-cxr-jpg
```

If these files and directories exist, the entrypoint skips downloading them.

### S3-Compatible Dataset Download

The Docker entrypoint can download missing dataset files and images with the AWS CLI. It uses `aws s3 --endpoint-url`, so it can target S3-compatible storage, not only AWS S3.

| Variable | Default | Used by | Description |
| --- | --- | --- | --- |
| `AWS_ACCESS_KEY_ID` | Empty in Compose | `awscli` | Access key for the S3-compatible endpoint. |
| `AWS_SECRET_ACCESS_KEY` | Empty in Compose | `awscli` | Secret key for the S3-compatible endpoint. |
| `DATASET_ENDPOINT_URL` | Empty | `entrypoint.sh` | S3-compatible endpoint URL. Required when files are not already mounted locally. |
| `DATASET_BUCKET` | `mmim` | `entrypoint.sh` | Bucket containing CSVs, stats, and images. |
| `DATASET_TRAINING_KEY` | `ds_train.csv` | `entrypoint.sh` | Object key for the training CSV. |
| `DATASET_VALIDATION_KEY` | `ds_val.csv` | `entrypoint.sh` | Object key for the validation CSV. |
| `DATASET_STATS_KEY` | `stats.json` | `entrypoint.sh` | Object key for dataset statistics. |
| `DATASET_IMAGES_DIR` | `mimic-cxr-jpg` | `entrypoint.sh`, `docker-compose.yml` | Object prefix for image files. |
| `DATASET_TRAINING_FILENAME` | Basename of `DATASET_TRAINING_KEY` | `entrypoint.sh` | Local filename used under `DATASET_LOCAL_DIR`. |
| `DATASET_VALIDATION_FILENAME` | Basename of `DATASET_VALIDATION_KEY` | `entrypoint.sh` | Local filename used under `DATASET_LOCAL_DIR`. |
| `DATASET_STATS_FILENAME` | Basename of `DATASET_STATS_KEY` | `entrypoint.sh` | Local filename used under `DATASET_LOCAL_DIR`. |

Example:

```bash
DATASET_ENDPOINT_URL=https://s3.example.com \
DATASET_BUCKET=mmim \
AWS_ACCESS_KEY_ID=... \
AWS_SECRET_ACCESS_KEY=... \
docker compose up --build trainer
```

### MLflow

MLflow settings are passed through Docker Compose and read by MLflow itself or by training code.

| Variable | Default | Used by | Description |
| --- | --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `http://host.docker.internal:5000` in Compose | MLflow | Tracking server URI. |
| `MLFLOW_TRACKING_USERNAME` | `admin` in Compose | MLflow | Tracking server username, if authentication is enabled. |
| `MLFLOW_TRACKING_PASSWORD` | `password1234` in Compose | MLflow | Tracking server password, if authentication is enabled. |
| `MLFLOW_EXPERIMENT_NAME` | `Multimodal ICU mortality` | `trainer/train.py` | Experiment name used for training runs. |

Training metadata logged to MLflow includes source revision, dataset hashes, dataset paths, selected hyperparameters, platform, Python, PyTorch, TorchVision, CUDA version, and GPU name.

### Training Hyperparameters

These are read by `trainer/config.py` at Python runtime.

| Variable | Default | Type | Description |
| --- | --- | --- | --- |
| `MMIM_BATCH_SIZE` | `32` | int | Batch size. |
| `MMIM_EPOCHS` | `10` | int | Number of training epochs. |
| `MMIM_DROPOUT` | `0.3` | float | Dropout used by the fusion model. |
| `MMIM_LEARNING_RATE` | `0.001` | float | AdamW learning rate. Compose currently defaults to `0.0001`. |
| `MMIM_TRAIN_LIMIT` | `1.0` | float | Fraction of train and validation datasets to sample for faster iteration. |
| `MMIM_NUM_WORKERS` | Derived from CPU count | int | DataLoader worker count. |
| `MMIM_DEBUG` | `false` | bool | Enables dataset debug output. |
| `MMIM_DATASET_SHUFFLE` | `true` | bool | Enables DataLoader shuffling. |
| `MMIM_LOSS_POS_WEIGHT` | `5160 / 936` | float | Positive class weight for imbalanced mortality labels. Update if the training distribution changes. |

Boolean variables accept `1`, `true`, `yes`, and `on` as true values where parsed through `bool_from_env`.

## Running Training

Run training locally from the repository root:

```bash
uv run python -m trainer.main train
```

Run the container with Docker Compose:

```bash
docker compose up --build trainer
```

The container entrypoint resolves dataset paths, downloads missing inputs if configured, changes to `BASE_DIR`, and currently runs:

```bash
uv run python -m trainer.main
```

If `trainer.main` requires an explicit command, update `entrypoint.sh` to pass it, for example `uv run python -m trainer.main train`.

## Building Datasets

Dataset generation is implemented in `generator/builder.py` and writes output paths based on:

- `TRAINING_DATASET_FILE`
- `VALIDATION_DATASET_FILE`
- `TEST_DATASET_FILE`
- `DATASET_STATS_FILE`

When these variables are not set, builder defaults write into the current directory.

The builder expects:

- A DuckDB database containing MIMIC-IV and MIMIC-ED tables.
- A MIMIC-style image base directory.
- The MIMIC-CXR metadata CSV shipped with MIMIC-CXR-JPG.

## Limitations

- Data is sparse because the project uses ICU stays while looking back up to 24 hours before ICU admission.
- The model currently trains on one image per selected stay, so it does not model time trends across multiple images.
- Tabular time-series data is flattened into values such as `_min`, `_max`, and `_mean`.
