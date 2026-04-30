from typing import Any
import pandas as pd
import git
from git import Repo
import hashlib
from pathlib import Path


def find_paths(paths: list[str]) -> list[str]:
    return [p for p in paths if not Path(p).exists()]


def sha256str(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_local_repo() -> Repo:
    try:
        r = Repo(".")
    except git.exc.InvalidGitRepositoryError:
        raise ValueError(
            "Repo-related environment variables not found and this is not a git repo, please set GIT_SHA and GIT_REF or version this code."
        )
    return r


def df_schema(
    df: pd.DataFrame, label_column: str, id_columns: list[str]
) -> dict[str, Any]:
    columns = {}

    for col in df.columns:
        if col == label_column:
            role = "label"
        elif col in id_columns:
            role = "id"
        else:
            role = "feature"
        columns[col] = {"dtype": df[col].dtype.name, "role": role}
    return columns


def dataset_summary(ds: pd.DataFrame, label_column: str):
    total = len(ds)
    positives = int(ds[label_column].sum())
    negatives = int(total - positives)
    return {
        "total": total,
        "positives": positives,
        "negatives": negatives,
        "prevalence": positives / total,
    }


def leakage_check(train_ds: pd.DataFrame, val_ds: pd.DataFrame, test_ds: pd.DataFrame):
    train_set = set(train_ds["subject_id"])
    val_set = set(val_ds["subject_id"])
    test_set = set(test_ds["subject_id"])

    no_leakages = {
        "train_val_are_disjoint": train_set.isdisjoint(val_set),
        "train_test_are_disjoint": train_set.isdisjoint(test_set),
        "val_test_are_disjoint": val_set.isdisjoint(test_set),
    }

    if not all(no_leakages.values()):
        raise ValueError(f"Leakage tests not passed; leakage detected: {no_leakages}")

    return no_leakages
