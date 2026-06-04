# MMIM Generator

This module builds the dataset used by the multimodal ICU mortality prediction model.

In short it reads data from a DuckDB database and chest X-ray metadata to compute a cohort. It writes train, validation, and test CSVs, training statistics (mean and standard deviation for the training split), a schema file, and a `manifest.json` describing the dataset. An image directory must be provided, which is used to check that the selected images exist and to copy or upload them under the dataset prefix. Current supported formats for images (inferred by the code; mixed image formats in one dataset are not supported) are `jpeg` and `dicom` (extensions: `jpg`, `jpeg`, `dicom`, `dcm`).

The generator always writes a complete local dataset bundle under `--output-dir`. If LakeFS is configured, the same bundle is also uploaded to LakeFS and committed. In that case, the manifest records LakeFS as the canonical storage backend, while the local output remains a staging/debug copy.

## Requirements

The **bare minimum** requirements are the following:

* **MIMIC-IV** dataset: this is the primary data source.
* **MIMIC-ED** dataset: since we're focusing on the 24 hours preceding the ICU admission much data, unavailable in the base MIMIC-IV, will be extracted from this subsidiary database. As per documentation it refers to the same identifiers for patients, stays, lab values and other data.
* **MIMIC-CXR-JPG**: This is the primary images source for this project since working with 12-16 bits DICOM images is beyond the current possibilities. Importantly **even if you can work with full DICOM images, you need the `mimic-cxr-X.X.X-metadata.csv` file (only available under this dataset), which contains metadata valid for the MIMIC-CXR dataset as well and allows to skip computing it from raw images**.

**Optional** dependencies are:

* **MIMIC-CXR**: same content as MIMIC-CXR-JPG but images are available as full-depth DICOM files. **This project doesn't have support for this yet**.

## Usage

From the repository root:

```bash
uv run mmim-generator build-dataset \
  --database-path /path/to/mimic.duckdb \
  --metadata-file /path/to/mimic-cxr-2.0.0-metadata.csv.gz \
  --images-base-dir /path/to/mimic-cxr-jpg@mimic-cxr-jpg \
  --output-dir out
```

The **`--images-base-dir`** flag value is composed by

* a `<path>`: the path to the directory containing the images. This directory **must preserve the structure of the MIMIC-CXR/MIMIC-CXR-JPG repo and serve the patients prefix directory at its root**: an `ls` in that directory must show p10 to p19 for example. This is because path discovery follows the following template: `p{subject_prefix}/p{subject_id}/s{study_id}/{dicom_id}.{images_extension}`, where all except the latter come from MIMIC-IV.

* a separator: `@`.

* an `<alias>`: this is the name of the directory that will contain the images in the stores. It is not optional.

## Flags

| Flag | Required | Default | Description |
|---|---:|---|---|
| `-d`, `--database-path`, `--db`, `--database`, `--db-path` | yes | | Path to the DuckDB database. |
| `-m`, `--metadata-file`, `--metadata`, `--xcr-metadata-file` | yes | | Path to the MIMIC-CXR metadata file. |
| `-i`, `--images-base-dir`, `--images-basedir`, `--images-dir` | yes | | Image root plus alias, formatted as `<dir>@<alias>`. The path and alias must both be non-empty. |
| `-o`, `--output-dir`, `--out-dir`, `--out` | no | `out` | Local output directory. Can be overridden by `MMIM_GENERATOR_OUTPUT_DIR`. |
| `-w`, `--max-workers`, `--workers` | no | half of available CPUs | Number of parallel workers used for LakeFS image uploads. |
| `--debug` | no | `False` | Enables verbose store and upload logging. |

## Environment Variables

| Variable | Required | Description |
|---|---:|---|
| `GIT_SHA` | no | Generator code commit SHA. If omitted, the local git repository is inspected. |
| `GIT_REF` | no | Generator code branch or ref. If omitted, the local git repository is inspected. |
| `MMIM_GENERATOR_OUTPUT_DIR` | no | Overrides `--output-dir`. |
| `MMIM_GENERATOR_DATASET_VERSION` | no | Dataset version. Defaults to `v001` or the current default dataset version. |
| `MMIM_GENERATOR_BUILD_ID` | no | Build identifier used in LakeFS branch names. Defaults to a UTC timestamp + dataset version. |
| `LAKEFS_URL` | no | LakeFS URL. Enables LakeFS output when set. |
| `LAKEFS_ACCESS_KEY_ID` | yes, if LakeFS is enabled | LakeFS access key. Mandatory if LAKEFS_URL is set. |
| `LAKEFS_SECRET_ACCESS_KEY` | yes, if LakeFS is enabled | LakeFS secret key. Mandatory if LAKEFS_URL is set. |
| `LAKEFS_REPOSITORY` | yes, if LakeFS is enabled | LakeFS repository to write to. Mandatory if LAKEFS_URL is set. |

When LakeFS is enabled, the generator creates a build branch named like:

```text
build_<build_id>-<git_ref>-<git_sha_prefix>
```

`git_ref` has `/` and `_` replaced with `-`; `git_sha_prefix` is the first 9 characters of `GIT_SHA`. The default `build_id` is `YYYYMMDD_HHMM-<dataset_version>`, for example:

```text
build_20260604_1512-v001-orchestration-a1b2c3d4e
```

## Outputs

Local files written under the output directory:

```text
out/
├── manifest.json
├── info.txt
└── multimodal-icu-mortality-24h/
    └── <dataset_version>/
        ├── ds_train.csv
        ├── ds_val.csv
        ├── ds_test.csv
        ├── stats.json
        ├── schema.json
        └── <image_alias>/
            └── pXX/pXXXX/sXXXX/<dicom_id>.<ext>
```


Locally, `manifest.json` is written at the output root. In LakeFS, `manifest.json` is written both at the repository root and under the dataset prefix. Tabular files and images are placed under the dataset prefix.

| File | Description |
|---|---|
| `ds_train.csv` | Training split. |
| `ds_val.csv` | Validation split. |
| `ds_test.csv` | Test split. |
| `stats.json` | Mean and standard deviation values computed from the training split. |
| `schema.json` | Column schema and column roles. |
| `manifest.json` | Dataset metadata, source queries, split summaries, file paths, hashes, and defaults. |
| `info.txt` | Local commit-style metadata for the build. |

If LakeFS is enabled, the same dataset artifacts are written there as well. Generated CSV, JSON, and manifest files overwrite existing paths on the build branch; this does not mutate the branch the build was created from. Images are uploaded in parallel and skipped if the target path already exists.

> [!WARNING]
> **LakeFS image matching is path-based**. If a path already exists, the generator assumes the object is already present; it **does not hash or compare image contents**.

```json
{
  "manifest_version": "v1",
  "dataset": "multimodal-icu-mortality-24h",
  "dataset_version": "v001",
  "data_prefix": "multimodal-icu-mortality-24h/v001",
  "data": {
    "tabular": {
      "storage": {
        "kind": "lakefs",
        "repo": "mmim",
        "ref": "build_20260604_1512-v001-orchestration-a1b2c3d4e"
      },
      "extension": "csv",
      "label_column": "hospital_expire_flag",
      "files": {
        "training": {
          "format": "csv",
          "path": "ds_train.csv",
          "sha256": "..."
        },
        "validation": {
          "format": "csv",
          "path": "ds_val.csv",
          "sha256": "..."
        },
        "test": {
          "format": "csv",
          "path": "ds_test.csv",
          "sha256": "..."
        },
        "statistics": {
          "format": "json",
          "path": "stats.json",
          "sha256": "..."
        },
        "schema": {
          "format": "json",
          "path": "schema.json",
          "sha256": "..."
        }
      }
    },
    "images": {
      "storage": {
        "kind": "lakefs",
        "repo": "mmim",
        "ref": "build_20260604_1512-v001-orchestration-a1b2c3d4e"
      },
      "prefix": "mimic-cxr-jpg",
      "extension": "jpg",
      "path_template": "p{subject_prefix}/p{subject_id}/s{study_id}/{dicom_id}.{images_extension}"
    }
  }
}
```

The same manifest shape is used for filesystem output, with storage specs like:

```json
{
  "kind": "filesystem",
  "root": "out/"
}
```

Tabular data and images each carry their own storage spec, so they can be resolved independently by downstream code.


## Notes

The train, validation, and test split uses a fixed random seed.
