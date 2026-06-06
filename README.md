# In-Hospital Multimodal ICU Mortality Prediction in a MLOps Pipeline

This project scaffolds a prediction model in a MLOps pipeline.
The model is built to compare multimodal Vs tabular-only data performance at predicting a patient's death probability during the current stay at the moment of ICU admission. The pipeline goes from the dataset generation to (currently) tagging the best model as a `candidate` once quality gates have passed.

The model is NOT meant for production, this is **currently a learning project** that's all of the following:

* A way to put into practice as many new learnings I did in my Machine Learning learning path as possible.
* A way to learn MLOps coming from DevOps: I'm not focusing on Docker, Kubernetes deployment, GCP/AWS, Prometheus + Grafana (yet) since that would distract my attention from what I need to learn from scratch; the philosophy here is "to boldly go where I've never gone before", I'll later meet what I'm already familiar with halfway.
* A Python refresh.
* A way to figure out how far my knowledge is from what production needs and a map I'll use to fill the gap.
* An experiment to use AI models and agents responsibly and in a fully controlled way: not a replacement to writing/understanding the codebase, only code mentoring and learning augmentation (and maybe writing some tests). Learning resources are still books and courses with the difference that topics can be identified in a more focused way and priorities can be better planned.

A small disclaimer: although I used to attend Medical School and was close to graduating when I had to leave, intensive care and its recent findings and literature are something I can't say I'm familiar with, so should this project reach a good level from other point of views, it would at least need a specialist to bring it closer to production.

## Model

The main question the model tries to answer is "Does providing chest X-Ray images along with demographics, vitals, and lab test data to a model improve the prediction accuracy for a model that has to assess the death probability within the current patient stay?".
The basic intent is provide a sharper tool when it comes to deciding what care pathway, people and tools to put in place to get the best outcome; the model should ideally, for example, help clinicians/triage determine what patient needs closer monitoring or more experienced nurses.

This project estimates in-hospital death risk for patients **just admitted** to the ICU. It uses data from the 24 hours before ICU admission up to ICU admission time, where `T=0` is `mimiciv_icu.icustays.intime`.

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

### Model Architecture

The model is a composite neural network with:

- A *visual encoder* for chest X-ray image features.
- A *tabular encoder* for vitals, labs, demographics, and missingness indicators.
- A *fusion* model that combines both representations to predict in-hospital mortality.

The Fusion model still needs to prove that multimodal learning helps. Planned baselines will compare it against image-only CNN binary classification and tabular-only Decision Tree/Logistic Regression binary classification to determine whether fusion is an improvement, a performance penalty, or no meaningful difference.

For more information about the model's intent, architecture and current limitations see the [Trainer module](src/mmim/trainer/README.md).

## Pipeline

This project is based on MLflow + Dagster. MLflow can be externally deployed and Dagster can be run with `dg dev` or with `dgd` after sourcing `./up.sh`.

Overall steps of the pipeline:

1. Generate the dataset (optional if already built) + manifest. See the [Generator module](src/mmim/generator/README.md) for more info about the manifest definition and for instructions on how to perform this step.
2. Train the model. The MLflow instance must be available. See the [Trainer module](src/mmim/trainer/README.md) for more info about how to run it.
3. Evaluate and promote the model to a `candidate` if gates pass. See the [Orchestrator module](src/mmim/orchestrator/README.md) to learn more on how to get started, what is the orchestration rationale and what gates are in place.

The pipeline is designed to accommodate for runs that happen outside the orchestrator, which can still provide a better model than the current one being tested and all the orchestrated ones. This means that the MLflow instance can be pointed to by runs without the need for the orchestrator to be aware of it. At some point, however, if you want the model to progress the orchestrator's quality gate asset has to run.

## Main Features

* **Data as a first-class citizen**: ML does not just require code to run, a change in training data can make a big difference, that's why a relevant portion of the code is dedicate to data versioning (LakeFS + Query hashing) and model lineage so it's always visible what data a model was trained on.
* **Manifest-driven dataset contract** with filesystem/LakeFS storage support: a dataset of images and tabular data is always accompanied by a detailed manifest to be able to rebuild data entirely.
* **Dagster orchestration** for dataset generation, training, model quality gates and model alias promotion.
* **MLflow experiment tracking and model registry** integration with tracking of code commits the model came from.
* **Multimodal PyTorch model** with tabular and CXR image branches.
* **Nix-based** environment for greater reproducibility.

> Note: not using tools like `uv2nix` to lock python dependencies yet but the good old `uv.lock` is there to help for the moment.

## Documentation

| Area | Docs |
|---|---|
| Dataset generation and manifest format | [Generator module](src/mmim/generator/README.md) |
| Model architecture, training, metrics, Grad-CAM | [Trainer module](src/mmim/trainer/README.md) |
| Dagster assets and quality gates | [Orchestrator module](src/mmim/orchestrator/README.md) |
| Filesystem/LakeFS storage abstraction | [Store module](src/mmim/store/README.md) |

## Data And Privacy

This repository does not include MIMIC data, generated datasets, model artifacts, MLflow artifacts, or local environment files. Users must provide their own credentialed access to MIMIC-IV, MIMIC-ED, and MIMIC-CXR/MIMIC-CXR-JPG.

Do not commit generated datasets, images, credentials, `.env` files, MLflow artifacts, or other potentially sensitive local outputs.

## Quick Start

Before running commands, either make sure `uv` and the required native dependencies are available in your shell, or enter the project development environment first:

```bash
nix develop
```

Prerequisites:

- A running MLflow tracking server, unless the local sqlite fallback is enough for development.
- A generated `ManifestV1`, or the inputs needed to generate one.
- Credentialed access to the required MIMIC datasets. Data is of course not included in this repository.

Start Dagster from the repository root:

```bash
uv run --extra orchestrator dagster dev -m mmim.orchestrator.definitions
```

For a smoke run, materialize the Dagster assets with an existing manifest URI such as `file://out/manifest.json`. See the [orchestrator README](src/mmim/orchestrator/README.md) for the complete materialization config.

To generate a dataset directly:

```bash
uv run --extra generator mmim-generator build-dataset \
  --database-path /path/to/mimic.duckdb \
  --metadata-file /path/to/mimic-cxr-metadata.csv.gz \
  --images-base-dir /path/to/mimic-cxr-jpg@mimic-cxr-jpg \
  --output-dir out
```

To train directly from a manifest:

```bash
uv run --extra trainer mmim-trainer train --manifest-uri file://out/manifest.json
```

## Project Structure

```text
src/mmim/
├── generator/      # Builds dataset bundles and ManifestV1
├── trainer/        # Trains the current Fusion model
├── orchestrator/   # Dagster assets for generation, training, and quality gates
└── store/          # Filesystem/LakeFS storage abstraction

tests/
└── unit/           # Unit tests for trainer, store, dataset loading, and model pieces
```

Root-level files:

| Path | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, optional dependency groups, CLI scripts, and Dagster config. |
| `README.md` | High-level project overview. |

## Development

Run the test suite:

```bash
uv run pytest
```

Run a style check:

```bash
uv run ruff check src tests
```

Pre-commit hooks are in place.

## Current Status

Current features:

- Dataset generation to local filesystem and optionally LakeFS.
- Manifest-driven trainer input.
- Fusion model training.
- MLflow experiment tracking.
- Dagster quality gate assigning the `candidate` alias.

The model is being currently developed using MIMIC-IV, MIMIC-ED and MIMIC-CXR-JPG, **Dicom format support is not yet in place** although the code correctly valitates the `.dcm` and `.dicom` extensions.

## Roadmap (let's dream big)

* ~~Basic orchestration~~

* Baseline model (Decision Tree/Logistic Regression and CNN with binary classification) for standalone tabular-only and visual-only predictions.

* Full orchestration:
    1. Orchestration also compares to baseline model
    2. Model deploy (platform: to be defined)
    3. Model test (test set, shadow testing, Kubernetes + traffic mirroring?)
    4. Revert to previous model if performance on the test set is not better than that on the validation test, log situation

* Monitoring

* Dashboard + explanatory LLM (monitoring system connector) + trusted client-side LLM-specified frontend components

* GCP/AWS training

* Federated training

* Serving and ONNX/Burn inference (Maturin + PyO3 for Rust inference engine):
    1. Add `mmim-serve` to load promoted models from MLflow by alias.
    2. Export trained PyTorch models to ONNX and log them with `mlflow.onnx` (ONNX + pyfunc) in addition to using pytorch + pyfunc (current behavior).
    3. Add parity checks (quality gate!) between PyTorch, ONNX Runtime, and Burn inference: guarantees that we're running the intended model.
    4. Use Burn (ONNX) as an inference backend

  Python is the interface with the client and MLFlow, Rust is for the inference backend only.

  Advantages:
    * CPU inference performance gain (Maybe? needs benchmarking)
    * GPU inference performance gain (Maybe? needs benchmarking)
    * CPU/WebGPU/WASM portability
    * The inference runtime is more predictable, constrained and portable (no python dependencies for the inference backend)

* As long as the inference engine is a bare compiled executable:

	* CLI/Desktop App takes in data and runs inference, decoupling from python.
	* If a frontend is ever built and the model is small enough it could even run in the browser (if weights are open source) via WASM (WebGPU) without much coding effort, feasibility depends on model size, browser performance, preprocessing parity, and model artifact security.

## The Role of AI in this project

In a time where AI agents could complete my entire roadmap for this project in a single night and still have time to build a rocket before making me coffee, it was only used here for:

* Writing a fair amount of tests
* Validating software architectures and strategies
* Checking code refactoring
* Clarifying what topics to study in more depth before using them
* Fixing typos
* Writing the README files backbone I could iterate on
* Contain my humoristic vein

There's no point in rejecting this technology, neither is there in handing off coding completely to only focus on domain knowledge (which we could be tempted to hand off too): my opinion, backed from current research, is that we're not yet to a point where models + agents are proficient at writing code so, even I don't consider myself an expert python developer, it's still worth having complete awareness of the codebase instead of only inspecting problematic code when an issue arises.
