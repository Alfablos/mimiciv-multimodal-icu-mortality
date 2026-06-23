from mmim.generator.manifest import ManifestV1
import torchvision
import torch
import os
import mlflow
import mlflow.pytorch
import git
from git import Repo
import platform


from config import (
    dataset_shuffle,
    default_num_workers,
    num_workers,
)


def get_local_repo() -> Repo:
    try:
        r = Repo(".")
    except git.exc.InvalidGitRepositoryError:
        raise ValueError(
            "Repo-related environment variables not found and this is not a git repo, please set GIT_SHA and GIT_REF or version this code."
        )
    return r


def log_metadata(manifest: ManifestV1, no_send=False):
    git_sha = os.getenv("GIT_SHA")
    git_ref = os.getenv("GIT_REF")

    if git_sha is None or git_ref is None:
        repo = get_local_repo()

    if git_sha is None:
        git_sha = repo.head.commit.hexsha
    if git_ref is None:
        git_ref = repo.head.ref.name

    # with open(train_csv, "rb") as f:
    #     dataset_train_hash = sha256(f.read()).hexdigest()
    # with open(val_csv, "rb") as f:
    #     dataset_validation_hash = sha256(f.read()).hexdigest()
    # with open(dataset_stats_file, "rb") as f:
    #     dataset_stats_hash = sha256(f.read()).hexdigest()

    metadata = {
        "trainer.git_sha": git_sha,
        "trainer.git_ref": git_ref,
        # "dataset.train_filepath": train_csv,
        # "dataset.train_sha256": dataset_train_hash,
        # "dataset.validation_filepath": val_csv,
        # "dataset.validation_sha256": dataset_validation_hash,
        # "dataset.stats_sha256": dataset_stats_hash,
        # "dataset.images_extension": image_extension,
        # "dataset.loss_positive_weight": loss_pos_weight,
        # "dataset.images_base_dir": image_base_dir,
        "dataset.name": manifest.dataset,
        "dataset.version": manifest.dataset_version,
        "dataset.defaults": manifest.defaults,
        "dataset.generator.git_sha": manifest.generator_code.git_sha,
        "dataset.generator.git_ref": manifest.generator_code.git_ref,
        "dataset.tabular_data_spec": manifest.data.tabular,
        "dataset.image_data_spec": manifest.data.images,
        "dataset.shuffle": dataset_shuffle,
        "environment.default_num_workers": default_num_workers,
        "environment.num_workers": num_workers,
        "environment.platform": platform.platform(),
        "environment.python_version": platform.python_version(),
        "environment.torch_version": torch.__version__,
        "environment.torchvision_version": torchvision.__version__,
        "environment.cuda_version": torch.version.cuda or "N/A",
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            metadata[f"environment.gpu{i}_name"] = torch.cuda.get_device_name(device=i)

    if not no_send:
        mlflow.log_params(metadata)

    return metadata
