# MMIM Generator

This module builds the dataset used by the multimodal ICU mortality prediction model.

In short it reads data from a DockDB database and chest X-ray metadata to copmute a cohort. It writes train, validation, and test CSVs, training statistics (namely mean and std deviation for the training set), a schema file, and a `manifest.json` describing the dataset. If an image directory is provided, it also checks that the selected images exist and copies or uploads them to LakeFS (see below) under the dataset prefix.

Local output is always written. LakeFS output is enabled by setting `LAKEFS_URL` and the related LakeFS credentials, which are mandatory if the URL variable is set.

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

* a `<path>`: the path to the directory containing the images. This directory **must preserve the structure of the MIMIC-CXR/MIMIC-CXR-JPG repo and serve the patients prefix directory at its root**: an `ls` in that directory must show p10 to p19 for example. This is because path discovery follows the following template: `p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.{images_extension}`, where all accept the latter come from MIMIC-IV.

* a separator: `@`.

* an `<alias>`: this is the name of the directory that will contain the images in the stores. It is not optional.

## Inputs

## Flags

| Flag | Required | Default | Description |
|---|---:|---|---|
| `-d`, `--database-path`, `--db`, `--database`, `--db-path` | yes | | Path to the DuckDB database. |
| `-m`, `--metadata-file`, `--metadata`, `--xcr-metadata-file` | yes | | Path to the MIMIC-CXR metadata file. |
| `-i`, `--images-base-dir`, `--images-basedir`, `--images-dir` | no | | Image root plus alias, formatted as `<dir>@<alias>`. If omitted, image existence checks are skipped. |
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
        ├── manifest.json
        └── <image_alias>/
            └── pXX/pXXXX/sXXXX/<dicom_id>.<ext>
```


`manifest.json` is placed at both the output root and under the dataset prefix, this allows to **have a pointer to the latest dataset version in the root and the possibility to inspect previous ones**. Tabular files and images are placed under the dataset prefix.

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
> **LakeFS image matching is path-based**. If a path already exists, the generator assumes the object is already present, it **does not hash or compare image contents**.

```json
{
  "data": {
    "images": {
      "branch": "build_20260505_1147-v001-split-projects-e1d160b46",
      "extension": "jpg",
      "path_template": "p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.{images_extension}",
      "prefix": "multimodal-icu-mortality-24h/v001/mimic-cxr-jpg",
      "repo": "mmim",
      "storage": "lakefs"
    },
    "tabular": {
      "branch": "build_20260505_1147-v001-split-projects-e1d160b46",
      "extension": "csv",
      "files": {
        "schema": {
          "format": "json",
          "path": "schema.json",
          "sha256": "4a861673db572b47390d26151026898ced49508a5182ca3197f09a4354a6fd3f"
        },
        "statistics": {
          "format": "json",
          "path": "stats.json",
          "sha256": "532ec49e42f924cf9fe35b251a3e1be2bb685d2168466f70e54bd6deec227a24"
        },
        "test": {
          "format": "csv",
          "path": "ds_test.csv",
          "sha256": "7498053dde9d72a43a52109a37e665b8feed5351392eaf5eb65e76dad8692677"
        },
        "training": {
          "format": "csv",
          "path": "ds_train.csv",
          "sha256": "7d6346e365665efe9a03b2ad400253ba6f882de7b8c39de1506f3bb00ed65d7c"
        },
        "validation": {
          "format": "csv",
          "path": "ds_val.csv",
          "sha256": "ac012818792e38724554a8d94fc96b916eea32c2a5a179e006967feef00e8ffc"
        }
      },
      "prefix": "multimodal-icu-mortality-24h/v001",
      "storage": "lakefs"
    }
  },
  "dataset": "multimodal-icu-mortality-24h",
  "dataset_version": "v001",
  "defaults": {
    "loss_pos_weight": 5.369905956112853
  },
  "generator_code": {
    "git_ref": "split-projects",
    "git_sha": "e1d160b469f641ee7c5b76b651345f09f7672dc9"
  },
  "lookback_window_hours": 24,
  "prediction_time": "icu_intime",
  "queries": {
    "cohort_query": "<cohort query redacted>",
    "cohort_query_sha256": "cd77c2bb338ef61b78e9476a5ea54deee5fdb1cfdbfa1d801f155776b9e3df69",
    "features_query": "<features query redacted>",
    "features_query_sha256": "d6553098823afd826446f898eb01cf0f29ebfe98c26b712b81b6ba28aa519e42",
    "images_query": "<images query redacted>",
    "images_query_sha256": "166b13533d70b8847d92c365c1b73f11debaf8ab2d0780e7943cab2364bb88f8"
  },
  "schema_version": "v1",
  "sources": [
    "MIMIC-IV",
    "MIMIC-ED",
    "MIMIC-CXR",
    "MIMIC-CXR-JPG"
  ],
  "splits": {
    "leakage_checks": {
      "train_test_are_disjoint": true,
      "train_val_are_disjoint": true,
      "val_test_are_disjoint": true
    },
    "random_seed": 42,
    "strategy": "first_stay_per_subject_first_cxr_random_split",
    "test": {
      "negatives": 657,
      "positives": 105,
      "prevalence": 0.1377952755905512,
      "total": 762
    },
    "train": {
      "negatives": 5139,
      "positives": 957,
      "prevalence": 0.15698818897637795,
      "total": 6096
    },
    "validation": {
      "negatives": 659,
      "positives": 103,
      "prevalence": 0.13517060367454067,
      "total": 762
    }
  }
}
```


## Notes

The train, validation, and test split uses a fixed random seed.
