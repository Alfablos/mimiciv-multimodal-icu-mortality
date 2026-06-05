# MMIM Orchestrator

The orchestrator is a small Dagster layer that wires together dataset generation, model training, and model promotion checks and showcases a possible simple pipeline. It should stay thin: dataset storage is described by `ManifestV1` and resolved through `mmim.store`, while generator and trainer logic stays in their own modules.

Current pipeline assets:

| Asset | Purpose | Output |
|---|---|---|
| `dataset_manifest` | Gets a `ManifestV1` either by loading an existing manifest URI or by running the generator. | `DatasetManifestOutput` |
| `training_run` | Loads a `ParsedDataset` from the manifest and trains the model. | `TrainingResult` |
| `quality_gate` | Checks validation metrics and registers/aliases a candidate model in MLflow. | No material output |

## Full Pipeline

The full pipeline will eventually:

1. Generate the training, validation and test sets. (_current_)
2. Train the model and collect metrics from the validation set. (_current_)
3. Validate the model against a fixed metric value and other versions that have entered the model lifecycle (_current_) after validating the Fusion model against baseline ones (Decision Trees or Logistic Regression, _planned_). If validation passes the model is then assigned the **candidate** alias.
4. Deploy the model (_planned_)
5. Test the model against the test set and act consequently (_planned_)

## Run Dagster

From the repository root:

```bash
uv run --extra orchestrator dagster dev -m mmim.orchestrator.definitions
```

The project also has Dagster `dg` metadata in `pyproject.toml`, so `dg`-based workflows can load `mmim.orchestrator` as the project root module.

## Dataset Manifest Asset

`dataset_manifest` has two modes.

If `manifest_uri` is set, it loads an existing manifest through `manifest_from_uri(...)`. No generator build runs.

If `manifest_uri` is not set, it requires `database_path`, `metadata_file`, and `images_base_dir`, then calls `generator.builder.build(...)`. The generator always writes a local output bundle and may also upload to LakeFS depending on `LAKEFS_*` environment variables.

Config fields:

| Field | Required | Default | Description |
|---|---:|---|---|
| `manifest_uri` | no | `null` | Existing manifest URI, e.g. `file://out/manifest.json` or `lakefs://repo/ref/manifest.json`. |
| `database_path` | yes, if generating | `null` | DuckDB database path. |
| `metadata_file` | yes, if generating | `null` | MIMIC-CXR metadata file. |
| `images_base_dir` | yes, if generating | `null` | Image root plus alias, formatted as `<dir>@<alias>`. |
| `output_dir` | no | `./out` | Local generator output directory. |
| `max_workers` | no | `4` | Generator worker count. |
| `debug` | no | `false` | Debug logging flag. |

## Training Asset

`training_run` consumes `dataset_manifest`, converts the manifest to a `ParsedDataset`, and calls `start_train(...)` with explicit hyperparameters.

Config fields:

| Field | Default | Description |
|---|---|---|
| `working_directory` | `./out` | Local training working directory. For filesystem-backed datasets, trainer may use the dataset root directly. |
| `batch_size` | `32` | Dataloader batch size. |
| `epochs` | `1` | Number of training epochs. |
| `dropout` | `0.3` | Fusion model dropout. |
| `learning_rate` | `0.0001` | AdamW learning rate. |
| `train_limit` | `1.0` | Fraction of train/validation data to sample. Useful for smoke tests. |

## Quality Gate Asset

`quality_gate` consumes `training_run`. It checks that the run completed or was interrupted with a logged best model, applies a static metric threshold, compares against the best historical MLflow run, and assigns the MLflow `candidate` alias to the selected model version.

Config fields:

| Field | Default | Description |
|---|---|---|
| `model_selection_metric` | `MMIM_TRAINER_MODEL_SELECTION_METRIC` or `AUROC` | Metric used for model comparison. Must be one of `AUROC`, `AUPRC`, or `sens_at_95_spec`. |
| `AUROC` | `0.7` | Static AUROC threshold. |
| `AUPRC` | `0.5` | Static AUPRC threshold. |
| `sens_at_95_spec` | `0.7` | Static sensitivity-at-95%-specificity threshold. |
| `fake_pass` | `false` | Testing-only override that forces the gate to pass. |

`fake_pass=true` is only for orchestration smoke tests. It should not be used for real model promotion.

### Quality Gate Logic

The quality gate has two model-selection steps: a static gate and a dynamic gate.

The static gate asks whether the current run's best model is good enough in absolute terms. The selected `model_selection_metric` is read from the current run's best validation metrics, then compared with the corresponding configured threshold. For example, if `model_selection_metric` is `AUROC`, the current best model must have `AUROC >= config.AUROC`. If the static gate fails, no MLflow registry action is taken.

The dynamic gate asks whether the current run's best model is better than the best model already known to MLflow for the same experiment. It searches MLflow runs in the configured experiment with `tags.best_model.logged = 'true'`, then sorts by the validation metric that corresponds to `model_selection_metric`:

| `model_selection_metric` | MLflow sort metric |
|---|---|
| `AUROC` | `metrics.val_auroc DESC` |
| `AUPRC` | `metrics.val_auprc DESC` |
| `sens_at_95_spec` | `metrics.val_sens_at_95_spec DESC` |

The first run from that search is treated as the best historical run. If the current run's best model URI is the same as the best historical model URI, the current model remains the candidate. If another run has a better metric, that historical model URI becomes the candidate instead.

This means the dynamic gate compares against eligible MLflow runs, not only existing MLflow Registered Model versions. If someone runs the trainer outside Dagster and logs a better model in the same experiment with `best_model.logged=true`, a later `quality_gate` materialization can select that external run, register its model URI if needed, and move the `candidate` alias to it.

Outcome summary:

| Condition | Action |
|---|---|
| Training status is not `completed` or `interrupted` | Fail the asset. |
| Training produced no best model | Fail the asset. |
| Static gate fails | Return without changing the MLflow registry. |
| Static gate passes and current run is the best MLflow run | Register or reuse the current model, then assign `candidate`. |
| Static gate passes but another MLflow run is better | Register or reuse the historical best model, then assign `candidate`. |
| `fake_pass=true` | Treat the current run's best model as the candidate regardless of metrics. |

### Registered Model Behavior

The registered model name is derived from `MLFLOW_EXPERIMENT_NAME` through `experiment_family`, which lowercases the experiment name and replaces spaces with underscores. If the registered model does not exist, the gate creates it. The gate then searches registered model versions whose `source_path` matches the selected model URI. If a matching version exists, that version is reused. If not, a new model version is created from the selected model URI. Finally, the `candidate` alias is assigned to the selected version.

> [!WARNING]
> The dynamic gate uses MLflow run metrics, not only currently registered model versions. A model trained outside Dagster can become the selected candidate if it is logged in the same experiment with `best_model.logged=true` and has the best validation metric.

## Environment Variables

The orchestrator relies on the generator, trainer, and MLflow environment variables already used by those modules.

Generator/LakeFS variables:

| Variable | Description |
|---|---|
| `GIT_SHA`, `GIT_REF` | Optional generator provenance. If missing, the local git repo is inspected. |
| `MMIM_GENERATOR_OUTPUT_DIR` | Overrides generator `output_dir`. |
| `MMIM_GENERATOR_DATASET_VERSION` | Dataset version, defaulting to the generator default. |
| `MMIM_GENERATOR_BUILD_ID` | Build id used in LakeFS branch names. |
| `LAKEFS_URL` | Enables LakeFS upload/load when set. |
| `LAKEFS_ACCESS_KEY_ID`, `LAKEFS_SECRET_ACCESS_KEY`, `LAKEFS_REPOSITORY` | Required when LakeFS is enabled. |

Trainer/MLflow variables:

| Variable | Description |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow tracking server. Defaults to a local sqlite database in trainer code. |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | Optional MLflow credentials. |
| `MLFLOW_EXPERIMENT_NAME` | Experiment name. Defaults to `Multimodal ICU mortality`. |
| `MMIM_TRAINER_MODEL_SELECTION_METRIC` | Model selection metric. Defaults to `AUROC`. |
| `MMIM_TRAINER_DEBUG`, `MMIM_TRAINER_DATASET_SHUFFLE`, `MMIM_TRAINER_NUM_WORKERS` | Trainer runtime controls still read from the environment. |

## Test Materialization

Example test materialization config:

```yaml
ops:
  dataset_manifest:
    config:
      debug: false
      max_workers: 4
      output_dir: ./out
      manifest_uri: file://out/manifest.json
  training_run:
    config:
      batch_size: 32
      dropout: 0.3
      epochs: 1
      learning_rate: 0.0001
      # Portion of training/validation sets to use. Remember, if too low val_loss will be `NaN`!
      train_limit: 0.01
      working_directory: ./out
  quality_gate:
    config:
      AUPRC: 0.5
      AUROC: 0.7
      sens_at_95_spec: 0.7
      # set fake_pass to true to fake a better model than static and dynamic quality gates or lower the metric corresponding to model_selection_metric if the model performs bad (testing only)
      fake_pass: true
      model_selection_metric: AUROC
resources: {}
```

This config uses an existing local manifest, runs a short training pass, and forces the quality gate to pass. Use it only to test Dagster wiring and MLflow registration behavior.

## Notes

The old LakeFS `ConfigurableResource` is not part of the main pipeline path. LakeFS access should happen through manifest storage specs and `mmim.store`, not through orchestrator assets directly.

There is no separate evaluation asset yet. The current quality gate uses validation metrics produced during training.
