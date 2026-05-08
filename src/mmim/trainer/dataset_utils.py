from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse, unquote, ParseResult
from typing import Any
import json
import os
import pandas as pd
from io import BytesIO

from lakefs import Repository
from lakefs.client import Client as LakeFSClient

from mmim.store.base import ReadOnlyStore
from mmim.store.filesystem import FilesystemReadOnlyStore
from mmim.store.lakefs import LakeFSReadOnlyStore


@dataclass
class ParsedDataset:
    train_ds: pd.DataFrame
    val_ds: pd.DataFrame
    stats: dict[str, Any]
    defaults: dict[str, Any]
    data_prefix: str
    images_prefix: str
    images_extension: str
    images_path_template: str
    store: ReadOnlyStore
    manifest: dict[str, Any]


def parse_lakefs_uri(u: ParseResult) -> tuple[str, str, str]:
    repo = u.netloc
    other_parts = [unquote(p) for p in u.path.lstrip("/").split("/") if p]
    if len(other_parts) < 2:
        raise ValueError(
            "LakeFS URI parser needs at least 2 parts in the path of the URI for `ref` and `key`"
        )

    ref = other_parts[0]
    key = "/".join(other_parts[1:])

    return repo, ref, key


def store_from_manifest(manifest_path: str) -> tuple[dict[str, Any], ReadOnlyStore]:
    parsed_uri = urlparse(manifest_path)

    if parsed_uri.scheme == "file":
        path = manifest_path.removeprefix("file://")
        with open(path, "r") as m:
            manifest = json.load(m)

        store = FilesystemReadOnlyStore(
            f"{Path(path).parent}",
            prefix=manifest["data_prefix"],
        )

        return manifest, store

    elif parsed_uri.scheme == "lakefs":
        repo_id, ref_id, key = parse_lakefs_uri(parsed_uri)
        repo = Repository(
            repository_id=repo_id,
            client=LakeFSClient(
                host=os.getenv("LAKEFS_URL"),
                username=os.getenv("LAKEFS_ACCESS_KEY_ID"),
                password=os.getenv("LAKEFS_SECRET_ACCESS_KEY"),
            ),
        )
        ref = repo.ref(ref_id)
        manifest_obj = ref.object(key)
        if not manifest_obj.exists():
            raise ValueError(f"Couldn't find manifest at {manifest_path}")

        with manifest_obj.reader() as manifest_reader:
            manifest: dict[str, Any] = json.load(manifest_reader)

        store = LakeFSReadOnlyStore(ref=ref, prefix=manifest["data_prefix"])
        return manifest, store

    else:
        raise ValueError(
            f"Unrecognized prefix for manifest path: `{manifest_path}`. Allowed values are `file://` and `lakefs://`"
        )


def parse_manifest(
    manifest_path: str,
) -> ParsedDataset:
    manifest, store = store_from_manifest(manifest_path)
    data_prefix = manifest["data_prefix"]
    images_prefix = manifest["data"]["images"]["prefix"]
    images_extension = manifest["data"]["images"]["extension"]

    # TODO: instead of loading the whole dataset in memory use a reader in the store

    training_ds_str = store.read_bytes(
        manifest["data"]["tabular"]["files"]["training"]["path"]
    )  # defaults to with_prefix=True
    training_data_format: str = manifest["data"]["tabular"]["files"]["training"][
        "format"
    ]
    if training_data_format == "csv":
        train_ds = pd.read_csv(BytesIO(training_ds_str))
    else:
        raise ValueError(
            f"Unsupported data format for training set: {training_data_format}"
        )

    # TODO: compact

    validation_ds_str = store.read_bytes(
        manifest["data"]["tabular"]["files"]["validation"]["path"]
    )  # defaults to with_prefix=True
    validation_data_format: str = manifest["data"]["tabular"]["files"]["validation"][
        "format"
    ]
    if validation_data_format == "csv":
        val_ds = pd.read_csv(BytesIO(validation_ds_str))
    else:
        raise ValueError(
            f"Unsupported data format for validation set: {validation_data_format}"
        )

    training_stats_str = store.read_text(
        manifest["data"]["tabular"]["files"]["statistics"]["path"]
    )
    training_stats_format: str = manifest["data"]["tabular"]["files"]["statistics"][
        "format"
    ]
    if training_stats_format == "json":
        stats = json.loads(training_stats_str)
    else:
        raise ValueError(
            f"Unsupported data format for training stats: {training_stats_format}"
        )

    return ParsedDataset(
        train_ds=train_ds,
        val_ds=val_ds,
        stats=stats,
        defaults=manifest["defaults"],
        data_prefix=data_prefix,
        images_prefix=images_prefix,
        images_extension=images_extension,
        images_path_template=manifest["data"]["images"]["path_template"],
        store=store,
        manifest=manifest,
    )


if __name__ == "__main__":
    parse_manifest(
        "lakefs://mmim/build_20260506_1046-v001-split-projects-a7ce879a8/manifest.json"
    )
