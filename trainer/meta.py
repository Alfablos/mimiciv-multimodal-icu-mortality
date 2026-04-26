import os
import mlflow
import mlflow.pytorch
import git
from git import Repo
from hashlib import sha256


from .config import (
    train_csv,
    val_csv,
    image_extension,
    loss_pos_weight,
    dataset_stats_file,
    dataset_shuffle,
    default_num_workers,
    num_workers,
    image_base_dir,
)


def get_git_repo() -> Repo:
    try:
        r = Repo(".")
    except git.exc.InvalidGitRepositoryError:
        raise ValueError(
            "Repo-related environment variables not found and this is not a git repo, please set GIT_SHA and GIT_REF or version this code."
        )
    return r


def log_metadata(no_send=False):
    git_sha = os.getenv("GIT_SHA")
    git_ref = os.getenv("GIT_REF")

    if git_sha is None:
        repo = get_git_repo()
        git_sha = repo.head.commit.hexsha
    if git_ref is None:
        if not repo:
            repo = get_git_repo()
        git_ref = repo.head.ref.name

    with open(train_csv, "rb") as f:
        dataset_train_hash = sha256(f.read()).hexdigest()
    with open(val_csv, "rb") as f:
        dataset_validation_hash = sha256(f.read()).hexdigest()
    with open(dataset_stats_file, "rb") as f:
        dataset_stats_hash = sha256(f.read()).hexdigest()

    metadata = {
        "source.git_sha": git_sha,
        "source.git_ref": git_ref,
        "dataset.train_sha256": dataset_train_hash,
        "dataset.validation_sha256": dataset_validation_hash,
        "dataset.stats_sha256": dataset_stats_hash,
        "dataset.images_extension": image_extension,
        "dataset.loss_positive_weight": loss_pos_weight,
        "dataset.images_base_dir": image_base_dir,
        "dataset.shuffle": dataset_shuffle,
        "host.default_num_workers": default_num_workers,
        "host.num_workers": num_workers,
    }

    if not no_send:
        mlflow.log_params(metadata)

    return metadata
