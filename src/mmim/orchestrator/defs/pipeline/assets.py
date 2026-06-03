from mlflow.entities.model_registry import RegisteredModel
import os
from pathlib import Path
from multiprocessing import cpu_count
from mmim.orchestrator.defs.pipeline.model import DatasetManifestOutput
import dagster as dg
from mlflow import MlflowClient
from mlflow.entities import Run


from mmim.generator.builder import build, BuildOutput
from mmim.trainer.dataset_utils import manifest_from_uri, parsed_dataset_from_manifest
from mmim.trainer.train import start_train, TrainingResult
from mmim.trainer.config import (
    Hyperparameters,
    model_selection_metric,
    experiment_name,
    experiment_family,
)


from mmim.orchestrator.defs.pipeline.config import (
    DatasetManifestConfig,
    TrainingRunConfig,
    QualityGateConfig,
)

# TODO: mlflow client as a resource?


def _translate_metric() -> str:
    match model_selection_metric:
        case "AUROC":
            return "auroc"
        case "AUPRC":
            return "auprc"
        case "sens_at_95_spec":
            return "sens_at_95_spec"
        # should be unreachable because train.py checks this explicitly
        case _:
            raise ValueError(
                f"Metric `{model_selection_metric}` is not supported and cannot be translated"
            )


def _best_run(c: MlflowClient, exp_id: str) -> Run:
    runs = c.search_runs(
        experiment_ids=[exp_id],
        max_results=1,
        filter_string="tags.`best_model.logged` = 'true'",
        order_by=[f"metrics.val_{_translate_metric()} DESC"],
    )

    if len(runs) == 0:
        # The current training pipeline SHOULD have created a run, if it didn't it's a bug
        raise RuntimeError("Found no logged runs on MLFlow: this is a bug!")

    return runs[0]


def _ensure_registered_model(c: MlflowClient) -> RegisteredModel:
    registered_model_tags = {"domain": "intesive_care"}

    def _register_model() -> RegisteredModel:
        return c.create_registered_model(
            name=experiment_family, tags=registered_model_tags
        )

    registered_models = c.search_registered_models(
        filter_string=f"name = '{experiment_family}'"
    )

    if len(registered_models) == 0:
        registered_model = _register_model()
    else:
        registered_model = registered_models[0]

    for k, v in registered_model_tags.items():
        c.set_registered_model_tag(name=registered_model.name, key=k, value=v)

    return registered_model


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


# TODO: split into absolute gate (compare against QualityGateConfig) and relative gate (is this model better than the current candidate, staging, production existing aliases?)
@dg.asset
def quality_gate(
    context: dg.AssetExecutionContext,
    training_run: TrainingResult,
    config: QualityGateConfig,
):
    if config.fake_pass:
        context.log.warning(
            "fake_pass is set to True: the model will pass all the gates regardless of its performance. Be sure to know what you're doing."
        )

    training_status = training_run.train_results.train_status

    if training_status not in ["completed", "interrupted"]:
        raise ValueError(
            f"Cannot proceed to model evaluation because train status is `{training_status}`."
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

    if val < threshold:  # static gate
        # Do nothing, the model is no better than the ones we already have
        context.log.info(
            f"Model failed static quality gate: "
            f"{config.model_selection_metric}={current} < threshold={threshold}. "
            "No registry action will be taken."
        )
        return

    c = MlflowClient(
        tracking_uri=os.getenv(
            "MLFLOW_TRACKING_URI", "sqlite://" + str(Path.cwd() / "mlflow.db")
        )
    )
    experiment = c.get_experiment_by_name(experiment_name)

    if experiment is None:
        context.log.error(
            f"Experiment with name `{experiment_name}` not found. Aborting"
        )
        raise ValueError(
            f"Experiment with name `{experiment_name}` not found. Aborting"
        )

    experiment_id: str = experiment.experiment_id

    current_model_uri = training_run.train_results.best_model.uri
    best_run = _best_run(c, experiment_id)

    best_run_training_status: str = best_run.data.tags["training.status"]
    if best_run_training_status != "completed":
        context.log.warning(
            f"Run {best_run.info.run_id} was selected as current best run "
            f"but its training.status is reported as `{best_run_training_status}`."
        )

    best_model_uri = best_run.data.tags["best_model.model_uri"]

    have_switched_models = False

    if config.fake_pass:
        # fake_pass means: pretend the current model passed the improvement logic.
        # Do not switch to the historical best model while testing this path.
        target_model_uri = current_model_uri
        context.log.warning(
            f"fake_pass=True: treating current model `{current_model_uri}` "
            "as the new candidate regardless of its performance!"
        )

    elif current_model_uri != best_model_uri:
        # the current model performed no better than previously logged because sorting by best metric value doesn't return this model's run

        # switch!

        # Forget the current model, let's check that the best model is actually the candidate!
        target_model_uri = best_model_uri
        have_switched_models = True

    else:
        target_model_uri = current_model_uri

    if have_switched_models:
        context.log.info(
            f"Forget former current model, treating best model `{best_model_uri}` "
            "as the current since it performs better."
        )
    else:
        context.log.info(
            f"We have a new candidate: model `{target_model_uri}` is the best model so far."
        )

    # make sure the registered model exists and has proper tagging
    registered_model = _ensure_registered_model(c)

    # check if the current model uri already is in any version of the registered model
    # This is not mental onanism: this step does not protect against a model travelling to the past
    # rather, it covers orchestration retries, it allows to to reach some grade of idempotence if the
    # aliasing step fails and only the quality gate runs again
    previous_target_model_versions = c.search_model_versions(
        filter_string=(
            f"name = '{registered_model.name}' AND source_path = '{target_model_uri}'"
        )
    )

    already_registered = len(previous_target_model_versions) > 0

    if already_registered:
        # The version(s) this specific model was assigned in the past
        current_model_version = previous_target_model_versions[0].version
        context.log.info(
            f"Model `{target_model_uri}` has been registered in the past. "
            f"Reassigning old version {current_model_version}"
        )

    else:
        if have_switched_models:
            context.log.warning(
                f"Model `{target_model_uri}` is the best model so far but it's not "
                f"in any `{experiment_family}` version. Registering it now."
            )

        created_model_version = c.create_model_version(
            name=registered_model.name,
            source=target_model_uri,
        )

        current_model_version = created_model_version.version

        context.log.info(
            f"Model `{target_model_uri}` has never been registered before. "
            f"Assigning new version {current_model_version}"
        )

    c.set_registered_model_alias(
        name=registered_model.name,
        alias="candidate",
        version=current_model_version,
    )
