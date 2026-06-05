# MMIM Store

`mmim.store` is the small storage abstraction shared by the generator and trainer. It hides whether data is read from or written to the local filesystem or LakeFS, but it deliberately keeps path handling explicit.

The store layer is path-based. It does not track datasets, validate manifests, compare file contents, or perform automatic conflict resolution beyond the requested `overwrite` behavior.

## Store Types

Current read-only stores:

| Store | Backend | Purpose |
|---|---|---|
| `FilesystemReadOnlyStore` | Local filesystem | Reads text and bytes from an existing local directory. |
| `LakeFSReadOnlyStore` | LakeFS reference | Reads text and bytes from a LakeFS branch, tag, or commit reference. |

Current write-only stores:

| Store | Backend | Purpose |
|---|---|---|
| `FilesystemWriteOnlyStore` | Local filesystem | Writes text, bytes, and copied files under a local directory. |
| `LakeFSWriteOnlyStore` | LakeFS branch | Writes text, bytes, and copied files to a LakeFS branch. |

The common read API is:

| Method | Description |
|---|---|
| `read_text(path, with_prefix=True)` | Reads a text file. |
| `read_bytes(path, with_prefix=True)` | Reads a binary file. |
| `set_prefix(prefix)` | Changes the default prefix used for later operations. |

The common write API is:

| Method | Description |
|---|---|
| `write_text(path, data, overwrite=False, with_prefix=True)` | Writes a text object. |
| `write_bytes(path, data, overwrite=False, with_prefix=True)` | Writes a binary object. |
| `write_file(local_path, remote_path, overwrite=False, with_prefix=True)` | Copies a local file into the store. |
| `exists(path, with_prefix=True)` | Checks whether the target path exists in the store. |
| `commit(message, metadata={})` | Finalizes or records a write batch. |
| `set_prefix(prefix)` | Changes the default prefix used for later operations. |

## Prefixes And Paths

Stores resolve paths by combining a backend root or reference, an optional prefix, and the provided path.

Example filesystem write:

| Component | Value |
|---|---|
| Filesystem root | `out/` |
| Store prefix | `multimodal-icu-mortality-24h/v001` |
| Write path | `ds_train.csv` |
| Resolved file | `out/multimodal-icu-mortality-24h/v001/ds_train.csv` |

Example LakeFS write:

| Component | Value |
|---|---|
| Repository | `mmim` |
| Branch | `build_20260604_1512-v001-orchestration-a1b2c3d4e` |
| Store prefix | `multimodal-icu-mortality-24h/v001` |
| Write path | `ds_train.csv` |
| Resolved object | `multimodal-icu-mortality-24h/v001/ds_train.csv` on that branch |

By default, operations use the store prefix. Passing `with_prefix=False` disables the prefix for that operation and reads or writes at the backend root or LakeFS ref root.

## Overwrite Semantics

All write methods take an explicit `overwrite` flag. The default is `overwrite=False`.

| Condition | Behavior |
|---|---|
| Target does not exist | The write succeeds. |
| Target exists and `overwrite=False` | The store raises `FileExistsError`. |
| Target exists and `overwrite=True` | The store replaces the target content. |

This applies to `write_text`, `write_bytes`, and `write_file` for both filesystem and LakeFS stores.

Filesystem overwrite means replacing the local file at the resolved path. Parent directories are created as needed.

LakeFS overwrite means replacing the object on the write branch. The branch the build was created from is not mutated by the write; LakeFS changes become visible when the branch is committed.

`FilesystemWriteOnlyStore.commit(...)` writes an `info.txt` file at the filesystem store root with `overwrite=True`. `LakeFSWriteOnlyStore.commit(...)` creates a LakeFS commit on the write branch.

## Generator Write Policy

The generator uses the store API with two different policies.

Tabular and metadata artifacts are always rewritten:

| Artifact | Store operation |
|---|---|
| `ds_train.csv` | `write_text(..., overwrite=True)` |
| `ds_val.csv` | `write_text(..., overwrite=True)` |
| `ds_test.csv` | `write_text(..., overwrite=True)` |
| `stats.json` | `write_text(..., overwrite=True)` |
| `schema.json` | `write_text(..., overwrite=True)` |
| `manifest.json` | `write_text(..., overwrite=True)` |

The generator writes `manifest.json` under the dataset prefix and also writes a root-level copy with `with_prefix=False`. The root-level manifest is a convenience copy of the latest generated dataset version manifest for that filesystem output directory or LakeFS repository branch.

Images are not overwritten by the generator. For each selected image, the generator computes the image path relative to the input image base directory, switches the store prefix to `<data_prefix>/<image_alias>`, and checks whether that relative path already exists in the target store. If it exists, the image is skipped. If it does not exist, the image is copied or uploaded with `overwrite=False`.

## Image File Matching

> [!WARNING]
> Image matching is filename/path-based only. If the target image path exists, the generator treats the image as already present and skips it. The store does not compare image contents.

The image existence check uses the resolved target path, for example:

```text
multimodal-icu-mortality-24h/v001/mimic-cxr-jpg/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
```

No image content comparison is performed. The generator does not compare:

| Not checked | Consequence |
|---|---|
| SHA256 hash | A corrupted image with the same target path is treated as present. |
| File size | A truncated image with the same target path is treated as present. |
| Modification time | An older image with the same target path is treated as present. |
| Source file identity | A different local source image copied to the same target path is treated as present. |

This behavior is intentional for fast rebuilds and LakeFS uploads, but it means image refresh is manual. To force image replacement, remove the target image/object, use a different dataset version or image alias, write to a fresh LakeFS build branch, or change the write policy in the generator.

Tabular files and JSON metadata are different: their content hashes are recorded in `ManifestV1`. Image files currently do not have per-image hashes in the manifest.

## Manifest Relationship

The manifest describes which storage backend and prefix downstream code should use. `trainer.dataset_utils.store_from_storage(...)` converts a manifest storage spec into a read-only store.

For generated datasets, tabular data and image data each carry their own storage spec in `ManifestV1`. They usually point to the same backend, but downstream code should resolve them independently.

Typical manifest storage specs are:

```json
{
  "kind": "filesystem",
  "root": "out/"
}
```

```json
{
  "kind": "lakefs",
  "repo": "mmim",
  "ref": "build_20260604_1512-v001-orchestration-a1b2c3d4e"
}
```

The trainer reads tabular CSV and JSON files through the tabular store. It reads image bytes through the image store when it needs to populate the local training cache.

## Backend Notes

Filesystem behavior:

| Store | Behavior |
|---|---|
| `FilesystemReadOnlyStore` | Requires the root directory to already exist. Raises if the root is missing or is a file. |
| `FilesystemWriteOnlyStore` | Creates the root directory if needed. Raises if the root path already exists as a file. |

LakeFS behavior:

| Store | Behavior |
|---|---|
| `LakeFSReadOnlyStore` | Reads from a LakeFS `Reference`. The reference can point to a branch, tag, or commit. |
| `LakeFSWriteOnlyStore` | Creates or reuses a write branch from `base_branch`, defaulting to `master` when no base branch is supplied. |

## Current Limits

The store layer currently has no delete API, no list API, no content-addressed image storage, no automatic image integrity verification, and no automatic conflict resolution beyond the explicit `overwrite` flag.
