import os
from pathlib import Path
from multiprocessing import cpu_count
from mmim.orchestrator.defs.pipeline.model import DatasetManifestOutput
import dagster as dg
from mlflow import MlflowClient


from mmim.generator.builder import build, BuildOutput
from mmim.trainer.dataset_utils import manifest_from_uri, parsed_dataset_from_manifest
from mmim.trainer.train import start_train, TrainingResult
from mmim.trainer.config import Hyperparameters


from mmim.orchestrator.defs.pipeline.config import (
    DatasetManifestConfig,
    TrainingRunConfig,
    QualityGateConfig,
)

mlflow_experiment_name = "Multimodal ICU mortality"


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
        generator_build_output: BuildOutput = build(
            duckdb_db=config.database_path,
            metadata_file=config.metadata_file,
            images_base_dir=config.images_base_dir,
            max_workers=max(((cpu_count() or 1) // 2) - 2, 0),
            debug=True,
            output_dir="./out",
        )

        dataset_manifest_output = DatasetManifestOutput(
            source="generated",
            manifest=generator_build_output.manifest,
            manifest_uri=None,
            output_dir=generator_build_output.output_dir,
            lakefs_ref=generator_build_output.lakefs_ref,
        )

        context.add_output_metadata(dataset_manifest_output)

        return dataset_manifest_output


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

    context.add_output_metadata({"training_result": training_result.model_dump_json()})

    return training_result


@dg.asset
def quality_gate(
    context: dg.AssetExecutionContext,
    training_run: TrainingResult,
    config: QualityGateConfig,
):
    training_status = training_run.train_results.train_status
    if training_status not in ["completed", "interrupted"]:
        raise ValueError(
            f"Cannot proceed to model evaluation because train status is {training_status}"
        )

    assert training_run.train_results.best_model is not None, (
        "If training is completed or interrupted the field `best_metrics` cannot be None. This is a bug!"
    )
    assert training_run.train_results.best_model.metrics.AUROC is not None, (
        "If training is completed or interrupted the field `best_metrics.AUROC` cannot be None. This is a bug!"
    )
    assert training_run.train_results.best_model.metrics.AUPRC is not None, (
        "If training is completed or interrupted the field `best_metrics.AUPRC` cannot be None. This is a bug!"
    )
    assert training_run.train_results.best_model.metrics.sens_at_95_spec is not None, (
        "If training is completed or interrupted the field `best_metrics.sens_at_95_spec` cannot be None. This is a bug!"
    )

    context.log.info(
        f"[PRE-REGISTRATION] best_model_uri={training_run.train_results.best_model.uri}"
    )

    threshold = float(getattr(config, config.model_selection_metric))
    current = float(
        getattr(
            training_run.train_results.best_model.metrics, config.model_selection_metric
        )
    )

    val = threshold + 0.001 if config.fake_pass else current

    if val >= threshold:
        # quality gate passed: now let's if the model is better than any other.

        # 1. search for similar models in the current experiment, sort them by model_selection_metric
        # 2. IF the current score is better than the above register it, else do nothing

        mlflow_client = MlflowClient(
            tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI", "sqlite://" + str(Path.cwd() / "mlflow.db")
            )
        )
        experiment = mlflow_client.get_experiment_by_name(mlflow_experiment_name)

        if experiment is None:
            context.log.error(
                f"Experiment with name `{mlflow_experiment_name}` not found. Aborting"
            )
            raise ValueError(
                f"Experiment with name `{mlflow_experiment_name}` not found. Aborting"
            )

        remote_models = []

        context.log.info("Searching for logged models...")
        remote_models_search = mlflow_client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"name = '{training_run.train_results.best_model.name}'",
        )

        while remote_models_search.token is not None:
            context.log.debug(
                f"token: {remote_models_search.token}\nresults: {','.join(remote_models_search.to_list())}\n"
            )
            remote_models.extend(remote_models_search)
            remote_models_search = mlflow_client.search_logged_models(
                experiment_ids=["Multimodal ICU mortality"],
                filter_string=f"name = '{training_run.train_results.best_model.name}'",
                page_token=remote_models_search.token,
            )
        remote_models.extend(remote_models_search)

        context.log.debug(f"Models: {remote_models}")
    else:
        # Do nothing, the model is no better than the ones we already have
        pass
