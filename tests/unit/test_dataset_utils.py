import json

import pandas as pd

from mmim.generator.manifest import (
    DataFileSpecV1,
    DataSpecV1,
    DatasetDefaultsV1,
    FilesystemStorage,
    GeneratorCodeSpec,
    ImageDataSpec,
    LeakageCheckV1,
    ManifestV1,
    QueriesSpecV1,
    SplitSummaryV1,
    SplitsSpecV1,
    TabularDataSpec,
    TabularFilesSpecV1,
)
from mmim.store.filesystem import FilesystemReadOnlyStore
from mmim.trainer.dataset_utils import (
    ParsedDataset,
    manifest_from_uri,
    parsed_dataset_from_manifest,
)


def test_manifest_uri_then_parsed_dataset_loads_tabular_data_and_image_metadata(
    tmp_path,
):
    data_prefix = "multimodal-icu-mortality-24h/v001"
    dataset_dir = tmp_path / data_prefix
    dataset_dir.mkdir(parents=True)

    train_df = pd.DataFrame(
        [{"subject_id": 1, "study_id": 2, "dicom_id": "a", "hospital_expire_flag": 0}]
    )
    val_df = pd.DataFrame(
        [{"subject_id": 3, "study_id": 4, "dicom_id": "b", "hospital_expire_flag": 1}]
    )
    test_df = pd.DataFrame(
        [{"subject_id": 5, "study_id": 6, "dicom_id": "c", "hospital_expire_flag": 0}]
    )
    stats = {"mean": {"age": 10.0}, "std": {"age": 2.0}}

    train_df.to_csv(dataset_dir / "ds_train.csv", index=False)
    val_df.to_csv(dataset_dir / "ds_val.csv", index=False)
    test_df.to_csv(dataset_dir / "ds_test.csv", index=False)
    (dataset_dir / "stats.json").write_text(json.dumps(stats))
    (dataset_dir / "schema.json").write_text(json.dumps({}))

    storage = FilesystemStorage(kind="filesystem", root=str(tmp_path))
    split_summary = SplitSummaryV1(total=1, positives=0, negatives=1, prevalence=0.0)
    image_path_template = (
        "p{subject_prefix}/p{subject_id}/s{study_id}/{dicom_id}.{images_extension}"
    )
    manifest = ManifestV1(
        dataset="multimodal-icu-mortality-24h",
        dataset_version="v001",
        schema_version="v1",
        prediction_time="icu_intime",
        lookback_window_hours=24,
        sources=["MIMIC-IV", "MIMIC-CXR"],
        generator_code=GeneratorCodeSpec(git_sha="test", git_ref="test"),
        queries=QueriesSpecV1(
            images_query="select 1",
            cohort_query="select 1",
            features_query="select 1",
        ),
        splits=SplitsSpecV1(
            strategy="test",
            random_seed=42,
            train=split_summary,
            validation=split_summary,
            test=split_summary,
            leakage_checks=LeakageCheckV1(
                train_val_are_disjoint=True,
                train_test_are_disjoint=True,
                val_test_are_disjoint=True,
            ),
        ),
        data_prefix=data_prefix,
        data=DataSpecV1(
            tabular=TabularDataSpec(
                storage=storage,
                extension="csv",
                label_column="hospital_expire_flag",
                files=TabularFilesSpecV1(
                    training=DataFileSpecV1(
                        path="ds_train.csv", format="csv", sha256="test"
                    ),
                    validation=DataFileSpecV1(
                        path="ds_val.csv", format="csv", sha256="test"
                    ),
                    test=DataFileSpecV1(
                        path="ds_test.csv", format="csv", sha256="test"
                    ),
                    statistics=DataFileSpecV1(
                        path="stats.json", format="json", sha256="test"
                    ),
                    schema=DataFileSpecV1(
                        path="schema.json", format="json", sha256="test"
                    ),
                ),
            ),
            images=ImageDataSpec(
                storage=storage,
                prefix="mimic-cxr-jpg",
                extension="jpg",
                path_template=image_path_template,
            ),
        ),
        defaults=DatasetDefaultsV1(loss_pos_weight=5.0),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json", by_alias=True))
    )

    loaded_manifest = manifest_from_uri(f"file://{manifest_path}")
    parsed = parsed_dataset_from_manifest(loaded_manifest)

    assert isinstance(loaded_manifest, ManifestV1)
    assert loaded_manifest.data_prefix == data_prefix
    assert isinstance(parsed, ParsedDataset)
    assert isinstance(parsed.tabular_store, FilesystemReadOnlyStore)
    assert isinstance(parsed.images_store, FilesystemReadOnlyStore)
    assert parsed.train_ds.equals(train_df)
    assert parsed.val_ds.equals(val_df)
    assert parsed.test_ds.equals(test_df)
    assert parsed.stats == stats
    assert parsed.manifest.defaults.loss_pos_weight == 5.0
    assert parsed.manifest.data_prefix == data_prefix
    assert parsed.manifest.data.images.prefix == "mimic-cxr-jpg"
    assert parsed.manifest.data.images.extension == "jpg"
    assert parsed.manifest.data.images.path_template == image_path_template
