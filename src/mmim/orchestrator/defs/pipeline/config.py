from mmim.trainer.train import VALID_MODEL_SELECTION_METRICS
from mmim.trainer.config import model_selection_metric
import dagster as dg

if model_selection_metric not in VALID_MODEL_SELECTION_METRICS:
    raise ValueError(
        f"Invalid model_selection_metric={model_selection_metric}. "
        f"Expected one of: {', '.join(sorted(VALID_MODEL_SELECTION_METRICS))}"
    )


class DatasetManifestConfig(dg.Config):
    manifest_uri: str | None = None
    database_path: str | None = None
    metadata_file: str | None = None
    images_base_dir: str | None = None
    output_dir: str = "./out"
    max_workers: int = 4
    debug: bool = False


class TrainingRunConfig(dg.Config):
    working_directory: str = "./out"
    batch_size: int = 32
    epochs: int = 1
    dropout: float = 0.3
    learning_rate: float = 1e-4
    train_limit: float = 1.0


class QualityGateConfig(dg.Config):
    model_selection_metric: str = model_selection_metric
    AUROC: float = 0.7
    AUPRC: float = 0.5
    sens_at_95_spec: float = 0.7
    # Whether to pretend there's an improvement regardless of the improvement being present
    fake_pass: bool = False
