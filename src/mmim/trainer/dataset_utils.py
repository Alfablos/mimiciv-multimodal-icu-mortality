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
from mmim.generator.manifest import (
    ManifestV1,
    StorageSpec,
    LakeFSStorage,
    FilesystemStorage,
)


@dataclass
class ParsedDataset:
    train_ds: pd.DataFrame
    val_ds: pd.DataFrame
    test_ds: pd.DataFrame
    stats: dict[str, Any]
    manifest: ManifestV1
    tabular_store: ReadOnlyStore
    images_store: ReadOnlyStore


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


def load_manifest(manifest_uri: str) -> ManifestV1:
    parsed_uri = urlparse(manifest_uri)

    if parsed_uri.scheme == "file":
        path = manifest_uri.removeprefix("file://")
        with open(path, "r") as m:
            manifest_str = m.read()

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
            raise ValueError(f"Couldn't find manifest at {manifest_uri}")

        with manifest_obj.reader() as manifest_reader:
            manifest_str = manifest_reader.read()
            if isinstance(manifest_str, bytes):
                raise ValueError(
                    f"Unexpected type for {manifest_uri}: expected string, found bytes."
                )
            assert isinstance(manifest_str, str)
    else:
        raise ValueError(
            f"Unrecognized prefix for manifest path: `{manifest_uri}`. Allowed values are `file://` and `lakefs://`"
        )

    return ManifestV1.from_json(manifest_str)


def store_from_storage(storage: StorageSpec, prefix: str):
    match storage.kind:
        case "filesystem":
            assert isinstance(storage, FilesystemStorage)
            return FilesystemReadOnlyStore(dir=storage.root, prefix=prefix)
        case "lakefs":
            assert isinstance(storage, LakeFSStorage)
            repo = Repository(
                repository_id=storage.repo,
                client=LakeFSClient(
                    host=os.getenv("LAKEFS_URL"),
                    username=os.getenv("LAKEFS_ACCESS_KEY_ID"),
                    password=os.getenv("LAKEFS_SECRET_ACCESS_KEY"),
                ),
            )
            return LakeFSReadOnlyStore(ref=repo.ref(storage.ref), prefix=prefix)
        case _:
            raise ValueError(f"Wrong kind of storage: {storage.kind}. This is a bug.")


def parse_manifest(manifest_uri: str) -> ParsedDataset:
    # manifest, store = store_from_manifest(manifest_uri)

    manifest = load_manifest(manifest_uri)
    tabular_store = store_from_storage(
        storage=manifest.data.tabular.storage, prefix=manifest.data_prefix
    )

    images_store = store_from_storage(
        storage=manifest.data.images.storage,
        prefix=manifest.data_prefix + "/" + manifest.data.images.prefix,
    )

    # TODO: instead of loading the whole dataset in memory use a reader in the store

    training_ds_str = tabular_store.read_bytes(
        manifest.data.tabular.files.training.path
    )  # defaults to with_prefix=True
    training_data_format: str = manifest.data.tabular.files.training.format
    if training_data_format == "csv":
        train_ds = pd.read_csv(BytesIO(training_ds_str))
    else:
        raise ValueError(
            f"Unsupported data format for training set: {training_data_format}"
        )

    # TODO: compact

    validation_ds_str = tabular_store.read_bytes(
        manifest.data.tabular.files.validation.path
    )  # defaults to with_prefix=True
    validation_data_format: str = manifest.data.tabular.files.validation.format
    if validation_data_format == "csv":
        val_ds = pd.read_csv(BytesIO(validation_ds_str))
    else:
        raise ValueError(
            f"Unsupported data format for validation set: {validation_data_format}"
        )

    test_ds_str = tabular_store.read_bytes(
        manifest.data.tabular.files.test.path
    )  # defaults to with_prefix=True
    test_data_format: str = manifest.data.tabular.files.test.format
    if test_data_format == "csv":
        test_ds = pd.read_csv(BytesIO(test_ds_str))
    else:
        raise ValueError(f"Unsupported data format for test set: {test_data_format}")

    training_stats_str = tabular_store.read_text(
        manifest.data.tabular.files.statistics.path
    )
    training_stats_format: str = manifest.data.tabular.files.statistics.format
    if training_stats_format == "json":
        stats = json.loads(training_stats_str)
    else:
        raise ValueError(
            f"Unsupported data format for training stats: {training_stats_format}"
        )
    return ParsedDataset(
        manifest=manifest,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        stats=stats,
        tabular_store=tabular_store,
        images_store=images_store,
    )
