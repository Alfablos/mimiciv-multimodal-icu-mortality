import json

import pandas as pd

from mmim.store.filesystem import FilesystemReadOnlyStore
from mmim.trainer.dataset_utils import ParsedDataset, parse_manifest


def test_parse_manifest_loads_tabular_data_and_image_metadata(tmp_path):
    data_prefix = "multimodal-icu-mortality-24h/v001"
    dataset_dir = tmp_path / data_prefix
    dataset_dir.mkdir(parents=True)

    train_df = pd.DataFrame(
        [{"subject_id": 1, "study_id": 2, "dicom_id": "a", "hospital_expire_flag": 0}]
    )
    val_df = pd.DataFrame(
        [{"subject_id": 3, "study_id": 4, "dicom_id": "b", "hospital_expire_flag": 1}]
    )
    stats = {"mean": {"age": 10.0}, "std": {"age": 2.0}}

    train_df.to_csv(dataset_dir / "ds_train.csv", index=False)
    val_df.to_csv(dataset_dir / "ds_val.csv", index=False)
    (dataset_dir / "stats.json").write_text(json.dumps(stats))

    manifest = {
        "data_prefix": data_prefix,
        "defaults": {"loss_pos_weight": 5.0},
        "data": {
            "tabular": {
                "label_column": "hospital_expire_flag",
                "files": {
                    "training": {"path": "ds_train.csv", "format": "csv"},
                    "validation": {"path": "ds_val.csv", "format": "csv"},
                    "statistics": {"path": "stats.json", "format": "json"},
                },
            },
            "images": {
                "prefix": "mimic-cxr-jpg",
                "extension": "jpg",
                "path_template": "p{subject_prefix}/p{subject_id}/s{study_id}/{dicom_id}.{images_extension}",
            },
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    parsed = parse_manifest(f"file://{manifest_path}")

    assert isinstance(parsed, ParsedDataset)
    assert isinstance(parsed.store, FilesystemReadOnlyStore)
    assert parsed.train_ds.equals(train_df)
    assert parsed.val_ds.equals(val_df)
    assert parsed.stats == stats
    assert parsed.defaults == {"loss_pos_weight": 5.0}
    assert parsed.data_prefix == data_prefix
    assert parsed.images_prefix == "mimic-cxr-jpg"
    assert parsed.images_extension == "jpg"
    assert parsed.images_path_template == manifest["data"]["images"]["path_template"]
