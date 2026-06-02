from multiprocessing import cpu_count
from mmim.orchestrator.defs.pipeline.model import DatasetManifestOutput
import dagster as dg


from mmim.generator.builder import build
from mmim.trainer.dataset_utils import manifest_from_uri, parsed_dataset_from_manifest
from mmim.trainer.train import (
    start_train,
    TrainingResult,
    VALID_MODEL_SELECTION_METRICS,
)
from mmim.trainer.config import Hyperparameters, model_selection_metric


from mmim.orchestrator.defs.pipeline.config import (
    DatasetManifestConfig,
    TrainingRunConfig,
    QualityGateConfig,
)

if model_selection_metric not in VALID_MODEL_SELECTION_METRICS:
    raise ValueError(
        f"Invalid model_selection_metric={model_selection_metric}. "
        f"Expected one of: {', '.join(sorted(VALID_MODEL_SELECTION_METRICS))}"
    )


@dg.asset
def dataset_manifest(
    context: dg.AssetExecutionContext, config: DatasetManifestConfig
) -> DatasetManifestOutput:
    if config.manifest_uri is not None:
        manifest = manifest_from_uri(config.manifest_uri)
        return DatasetManifestOutput(
            source="existing",
            manifest=manifest,
            manifest_uri=config.manifest_uri,
            output_dir=config.output_dir,
        )
    else:
        if config.database_path is None:
            raise ValueError("database_path must be set if manifest_uri is not.")
        if config.metadata_file is None:
            raise ValueError("metadata_file must be set if manifest_uri is not.")
        if config.images_base_dir is None:
            raise ValueError("images_base_dir must be set if manifest_uri is not.")
        generator_build_output = build(
            duckdb_db=config.database_path,
            metadata_file=config.metadata_file,
            images_base_dir=config.images_base_dir,
            max_workers=max(((cpu_count() or 1) // 2) - 2, 0),
            debug=True,
            output_dir="./out",
        )
        manifest_uri = f"file://{generator_build_output.output_dir}/manifest.json"
        return DatasetManifestOutput(
            source="generated",
            manifest=generator_build_output.manifest,
            manifest_uri=manifest_uri,
            output_dir=generator_build_output.output_dir,
            lakefs_ref=generator_build_output.lakefs_ref,
        )


@dg.asset
def training_run(
    context: dg.AssetExecutionContext,
    dataset_manifest: DatasetManifestOutput,
    config: TrainingRunConfig,
) -> TrainingResult:
    ds_config = parsed_dataset_from_manifest(dataset_manifest.manifest)
    training_result = start_train(
        dataset_config=ds_config,
        hyperparameters=Hyperparameters(
            batch_size=config.batch_size,
            epochs=config.epochs,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            train_limit=config.train_limit,
        ),
        working_directory=config.working_directory,
    )
    return training_result


def quality_gate(
    training_run: TrainingResult,
    config: QualityGateConfig,
    fake_pass: bool = False,
    model_selection_metric: str = model_selection_metric,
):
    training_status = training_run.train_results.train_status
    if training_status not in ["completed", "interrupted"]:
        raise ValueError(
            f"Cannot proceed to model evaluation because train status is {training_status}"
        )

    assert training_run.train_results.best_metrics is not None, (
        "If training is completed or interrupted the field `best_metrics` cannot be None. This is a bug!"
    )
    assert training_run.train_results.best_metrics.AUROC is not None, (
        "If training is completed or interrupted the field `best_metrics.AUROC` cannot be None. This is a bug!"
    )
    assert training_run.train_results.best_metrics.AUPRC is not None, (
        "If training is completed or interrupted the field `best_metrics.AUPRC` cannot be None. This is a bug!"
    )
    assert training_run.train_results.best_metrics.sens_at_95_spec is not None, (
        "If training is completed or interrupted the field `best_metrics.sens_at_95_spec` cannot be None. This is a bug!"
    )

    if fake_pass:
        val = getattr(config.metrics, model_selection_metric)
        val += 0.01  # always pass the gate
    else:
        val = getattr(training_run.train_results.best_metrics, model_selection_metric)

    if val >= getattr(config.metrics, model_selection_metric):
        # gate passed
        pass
    else:
        pass
