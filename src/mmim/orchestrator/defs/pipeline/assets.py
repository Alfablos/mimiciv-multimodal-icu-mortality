from multiprocessing import cpu_count
from mmim.orchestrator.defs.pipeline.model import DatasetManifestOutput
import dagster as dg


from mmim.generator.builder import build
from mmim.trainer.dataset_utils import manifest_from_uri, parsed_dataset_from_manifest
from mmim.trainer.train import start_train, TrainingResult
from mmim.trainer.config import Hyperparameters


from mmim.orchestrator.defs.pipeline.config import (
    DatasetManifestConfig,
    TrainingRunConfig,
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
