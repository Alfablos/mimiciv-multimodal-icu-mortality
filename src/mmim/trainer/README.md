# MMIM Trainer

The trainer fits a binary classifier for in-hospital mortality at ICU admission. It consumes a `ManifestV1` dataset bundle produced by the generator, loads tabular clinical features and chest X-ray images, and trains the current multimodal Fusion model.

This README focuses on the machine learning setup. For orchestration and model promotion behavior, see [`mmim.orchestrator`](../orchestrator/README.md). For dataset storage semantics, see [`mmim.store`](../store/README.md).

## Current Goal

Given a patient path that's typically
```text
 ED | Surgery | Another Unit => ICU
```

**Assess IF the patient current stay death risk prediction can benefit from multimodality rather than being a conventional tabular-only (demographics, vitals, lab tests) one**.

Expected results (one of):

* Adding imaging information enhances prediction accuracy (See `Metrics` below).
* Adding imaging information adds more noise than beneficial information, accuracy is overall decreased.
* Adding imaging information doesn't significantly impact prediction accuracy.

For a more accurate evaluation the model should be trained and tested against the whole population and subgroups resulting from properly stratifying the former (_not currently implemented_).

## Current Fusion Model

The current model estimates in-hospital death risk using:

| Modality | Signal |
|---|---|
| Vision (Chest X-ray) | One selected image from the 24 hours before ICU admission. |
| Numeric (Tabular clinical features) | Demographics, labs, vitals, and missingness indicators from the 24-hour lookback window. |

The prediction target is `hospital_expire_flag` (the dataset generation algorithm already makes sure a single stay per patient is selected). The model outputs a single mortality logit per patient-stay row.

The current architecture is a late-fusion neural network:

```text
CXR image -> visual encoder -> image embedding
                                                \
                                                 concat -> fusion head -> mortality logit
                                                /
tabular features -> tabular encoder -> tabular embedding
```

Current components:

| Component | Current implementation |
|---|---|
| Visual encoder | Modified `torchvision` DenseNet121 with generic pretrained weights. By default, all DenseNet parameters are frozen except `denseblock4` and `norm5`; the classifier is replaced by an identity layer.
| Tabular encoder | Small MLP with batch normalization, ReLU, and dropout. Configurable first layer input to accommodate for variable input layer size.
| Fusion | Concatenation of image and tabular embeddings.
| Classification head | MLP ending in one output unit.

## Dataset Inputs

The trainer loads data according to `ManifestV1`, copying/downloading images from the store that are not found in the set working directory.

| Input | How it is used |
|---|---|
| Train CSV | Optimizes the model, tracks progress and training metrics are essential to evaluate overfitting with validation ones. |
| Validation CSV | Selects the best model and logs validation metrics. |
| Test CSV | Present in the bundle, but not currently used by the trainer loop. It will be used by the orchestrator to promote a `candidate` to a higher rank (_not currently implemented_). |
| `stats.json` | Provides train-derived means and standard deviations for tabular standardization. |
| Image storage spec | Resolves image bytes through `mmim.store` when local image files are missing. |
| Image path template | Maps each row to the selected CXR file. |

Tabular continuous features are standardized with train-set statistics. Binary variables, demographics, one-hot encoded variables and missingness indicators are left on their natural value.

Images are padded to square, resized to `512x512`, converted to floating point, and normalized with the image encoder's expected mean and standard deviation.

### Missingness Indicators

There usually is a missingness indicator when a feature has no value associated (the actual value is the median from the training set). This choice comes from the fact that I think it's useful for a model to correlate the patient condition with the facts that some, for example, lab tests were not performed rather than just filling with a default value (the model has no way to know if it's a default or the value was actually missing). Data can be unavailable for a number of reasons (not just a deliberate decision), the model should take that into account. For example:

* `glucose_*_missing`: maybe the patient was so critical that assessing glucose was not the point or asking the lab for it would have delayed more useful results. Deliberate.
* `lactate_*_missing`: maybe no sepsis/hypoperfusion/shock was suspected. Deliberate.
* The lab value exists but it wasn't taken/charted within the 24H preceding the ICU admission. Stochastic.
* `temp_f_missing`, `spO2_missing`, or `resp_rate_missing` can also depend by the setting (available machines) or care intensity. Stochastic.
* Missingness is still a signal if it reflects what the clinicians did NOT prioritize.
* Missingness is NOT a signal if it has nothing to do with clinical conditions/decision.

In general, this is a point that should very carefully be evaluated since it can be very ambiguous. Because at the same time the model could learn to _make no assumptions_ based on it.

Even **training a model without missingness indicators**, as long as enough training samples are left, **might decrease performance in a setting where data is missing**, because the model is not trained to think "Uhm, I cannot infer anything from this value" if values are always imputed.

## Training Objective

This is binary classification with **logits**, meaning that the model output is not a probability during training.

`BCEWithLogitsLoss` loss function combines sigmoid activation and binary cross-entropy in a numerically stable form, with the advantage of taking in a `pos_weight` parameter to handle **class imbalance**. Since in-hospital mortality is the positive class and (luckily!) it occurs way less frequently, this weight should represent the negative-to-positive ratio from the training split.

## Metrics And Model Selection

Validation metrics currently logged during training:

| Metric | Meaning |
|---|---|
| `AUROC` | Ranking performance across thresholds. |
| `AUPRC` | Precision-recall performance, especially important with mortality imbalance. |
| `sens_at_95_spec` | The model's sensitivity when only considering values corresponding to a specificity of at least 95% |

The best model is selected by `MMIM_TRAINER_MODEL_SELECTION_METRIC`, which must be one of `AUROC`, `AUPRC`, or `sens_at_95_spec`. The selected model is logged to MLflow and tagged so the orchestrator can later compare it against other candidate runs.

Current model selection uses validation metrics only. Final test-set evaluation, calibration checks, and deployment decisions are planned but not implemented in the trainer loop, they'll be used to further refine candidate or higher rank model aliases selection (_not currently implemented_)

## Grad-CAM

Grad-CAM is used as an inspection tool for the image branch. It highlights image regions that influenced the mortality logit for selected train and validation examples. "What **area** in this X-Ray most influenced your prediction?".

Current behavior:

| Item | Current setting |
|---|---|
| Supported model | Current Fusion model with DenseNet121 visual encoder. |
| Target layer | DenseNet121 `denseblock4`. |
| Logged examples | Up to three train and validation examples per epoch. |
| Destination | MLflow figure artifacts under `gradcam/epoch_<n>/...`. Grad-CAM images are stored in the Run only, not the model. |

Grad-CAM should be treated as a qualitative sanity check, not proof of clinical validity. A plausible heatmap does not prove causal reasoning, and an implausible heatmap is not the only way an image branch can fail:

* A model **sees an area, not a heart rather than a lung**: we cannot infer the reason why that area was selected without a good amount of uncertainty.
* A model can look at the right area for the wrong (unknown to us in a neural network) reason
* Looking at the wrong area might not be the only/main reason for a prediction failure, it wouldn't be correct to blame it for 100% of the error.

## MLflow

MLflow acts as a hardcoded experiment tracking and model registry backend.

The trainer emits:

* model weights
* metrics
* hyperparameters
* dataset provenance
* environment metadata
* Grad-CAM figures
* best-model tags, among which `best_model.logged`, used by the orchestrator to only select runs that logged a model.
* model artifacts
* the training status: interrupted runs may log models anyway but it's worth pointing out that there could still be room for improvement.


The orchestrator quality gate depends on MLflow run search, tags such as `best_model.logged=true`, model URIs, registered model versions, and aliases.

## Planned Baselines And Ablations to Evaluate Multimodality

The Fusion model must prove that combining modalities helps. Multimodal learning can improve performance, but it can also add noise, overfit, or dilute a stronger tabular signal.

Planned baselines:

| Experiment | Purpose |
|---|---|
| Tabular-only `DecisionTree` binary classifier | Simple nonlinear structured-data baseline. |
| Tabular-only `LogisticRegression` binary classifier | Simple linear structured-data baseline. |
| Image-only CNN binary classifier | Measures whether CXR images contain useful mortality signal for this target. |
| Tabular-only MLP | Checks whether the neural tabular branch can match simple sklearn baselines. |
| Fusion model | Tests whether both modalities together improve over the best unimodal model. |
| Fusion with shuffled images | Tests whether the image branch contributes real patient-specific signal. |
| Fusion with shuffled tabular features | Tests whether the tabular branch contributes real patient-specific signal. |

These experiments should use the same split, target, and metrics. Their goal is to determine whether Fusion is an actual improvement, a performance penalty, or no meaningful difference.

## Current Limitations

Current limitations to keep in mind:

* The visual encoder currently uses a **generic DenseNet121 model and weights**, which is both overkill and insufficient: if this were a task where images were a broad set of object DenseNet models are trained on that would be ok, but here the input is always the same: heart, lungs, medical devices, jewelry. At the same time a lung or an earring don't quite look like something a "vanilla" DenseNet could recognize, since they're seen through the eyes of a X-Ray image receptor, definitely something you don't train model against everyday. Solutions to the "but DenseNet is an incredibly good model!":
  * Retrain a DenseNet from scratch on X-Ray images. Good but:
    1. Hasn't anyone already done so? Yes, see next solution
    2. Do I have all that money? For compute and storage needed to train the model.
    3. I can't train the model on the same images the train, validation and test set come from, overfitting is going to be a big problem.
  * Use a model that's already trained on X-Ray, so the weights are already optimized for not caring about recognizing a car and can focus on all the ways a heart can appear in a X-Ray.

* The **temporal dimension** is stripped out: images are *limited to 1 per stay* (and for how the dataset is currently built this means 1 per patient, see the [Generator Module](../generator/README.md) and the generated dataset for a deep dive on DB queries used). This means that the patient progress is not taken into account: a sudden drop in vitals may have a different meaning than a progressive one, but the model doesn't care: it only looks at the closest diagnostics to the ICU admission. Tabular data over the preceding 24h are aggregated to a single view. This is because sometimes historical data is not available (think of a car accident for a patient that's never been seen in the current hospital) so having a model that performs well without historical data, while much harder, can be useful. However this wasn't actually a decision: training a model that evaluates 5D tensors (H x W x C x t x batch_size) where t is the number of time-sorted images for that patient would have been much more resource intensive than just using 1 image per stay. Architectural challenges would have arisen, like the fact that t should have been a fixed number for the training to not crash, but the simplest solution would have been to use a "temporal padding" after establishing a t: the model accepts t images per patient but I only have 1, then I provide 1 image + t-1 "black images" to act as a `missingness indicator` the model can learn to avoid inferring from; however, in a resource (money) constrained environment the necessary cloud resources to train the model were unreachable.

* **No diagnosis** taken into account: the primary danger I wanted to avoid is temporal leakage. ICU admission diagnosis may be formulated and charted post and for a model that's supposed to be running while the patient is entering the ICU this is not acceptable. More so about the diagnosis performed in the ICU, which happens way later in the process and is sometimes defined after the patient has left the ICU.

* **Few tabular features**: only (very) few features are currently being taken into account, they're way less than it takes to achieve a decent performance, more so if some are allowed to be missing. This is deliberate, at this stage of the project the bare minimum for a training loop is required. More clinical predictors will be added.

* **No calibration protocol**: "this won't work for you". At this stage of the project this is not supposed to be in place.

* **Fusion model is not guaranteed to outperform traditional models**: again, this is due to the stage of the project. The model must be compared against baseline models before there's an answer to the question. Additionally, no probabilistic safety measures (like Brier Score and Calibration curves) are in place, not even confidence intervals.

## Usage

Train from an existing manifest:

```bash
uv run mmim-trainer train --manifest-uri file://out/manifest.json
```

LakeFS manifests are also supported:

```bash
uv run mmim-trainer train --manifest-uri lakefs://<repo>/<ref>/manifest.json
```

### Configuration and Hyperparameters Tuning via environment variables

| Variable | Default | Description |
|---|---|---|
| `MMIM_TRAINER_BATCH_SIZE` | `32` | Training batch size. |
| `MMIM_TRAINER_EPOCHS` | `10` | Number of epochs. |
| `MMIM_TRAINER_DROPOUT` | `0.3` | Dropout used in the current Fusion model. |
| `MMIM_TRAINER_LEARNING_RATE` | `0.001` | AdamW learning rate. |
| `MMIM_TRAINER_TRAIN_LIMIT` | `1.0` | Fraction of train/validation data used for smoke tests. |
| `MMIM_TRAINER_MODEL_SELECTION_METRIC` | `AUROC` | Metric used to select the best model. |
| `MMIM_TRAINER_NUM_WORKERS` | CPU-derived | DataLoader worker count. |
| `MMIM_TRAINER_DATASET_SHUFFLE` | `true` | Whether to shuffle the training dataset. |
| `MLFLOW_TRACKING_URI` | local sqlite DB | MLflow tracking backend. |
| `MLFLOW_EXPERIMENT_NAME` | `Multimodal ICU mortality` | MLflow experiment name. |
| `MLFLOW_TRACKING_USERNAME` | None | MLflow basic auth username. |
| `MLFLOW_TRACKING_PASSWORD` | None | MLflow basic auth password. |

For Dagster execution, use the orchestrator pipeline instead of calling the trainer directly.
