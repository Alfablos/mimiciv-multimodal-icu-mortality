import pandas as pd
import git
from git import Repo
import hashlib
from pathlib import Path

from mmim.generator import manifest


def find_paths(paths: list[str]) -> list[str]:
    return [p for p in paths if not Path(p).exists()]


def sha256str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_images_extension(path: str, formats: list[str]) -> str | None:
    for format_ in formats:
        images = Path(path).glob(f"**/*.{format_}", case_sensitive=False)
        try:
            # Even one image in that format makes the function
            # return that format
            _ = next(images)
            return format_
        except StopIteration:
            continue

    return None


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
) -> manifest.SchemaSpecV1:
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


def compute_pos_negs(ds: pd.DataFrame, label_column: str) -> tuple[int, int, int]:
    total = len(ds)
    positives = int(ds[label_column].sum())
    negatives = int(total - positives)
    return total, positives, negatives


def dataset_summary(ds: pd.DataFrame, label_column: str) -> manifest.SplitSummaryV1:
    total, positives, negatives = compute_pos_negs(ds=ds, label_column=label_column)
    return manifest.SplitSummaryV1(
        total=total,
        positives=positives,
        negatives=negatives,
        prevalence=positives / total,
    )


def leakage_check(
    train_ds: pd.DataFrame, val_ds: pd.DataFrame, test_ds: pd.DataFrame
) -> manifest.LeakageCheckV1:
    train_set = set(train_ds["subject_id"])
    val_set = set(val_ds["subject_id"])
    test_set = set(test_ds["subject_id"])

    check = manifest.LeakageCheckV1(
        train_val_are_disjoint=train_set.isdisjoint(val_set),
        train_test_are_disjoint=train_set.isdisjoint(test_set),
        val_test_are_disjoint=val_set.isdisjoint(test_set),
    )

    if not check.is_passed():
        raise ValueError(f"Leakage tests not passed; leakage detected: {check}")

    return check


def build_image_paths(ds: pd.DataFrame, base_dir: str, images_extension):
    subject_ids = ds["subject_id"].astype(str)
    study_ids = ds["study_id"].astype(str)
    dicom_ids = ds["dicom_id"].astype(str)
    return (
        base_dir.rstrip("/")
        + "/p"
        + subject_ids.str[:2]
        + "/p"
        + subject_ids
        + "/s"
        + study_ids
        + "/"
        + dicom_ids
        + "."
        + images_extension.lstrip(".")
    )


def parse_ref_str(raw_ref: str) -> tuple[str, str]:
    _split = raw_ref.split("@")
    if len(_split) != 2:
        raise ValueError(
            f"Invalid syntax for provided ref `{raw_ref}`: valid format is <repo>@<ref>"
        )
    repo, ref = _split[0], _split[1]
    return repo, ref
