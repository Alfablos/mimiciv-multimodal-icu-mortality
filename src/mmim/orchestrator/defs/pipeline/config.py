import dagster as dg


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
