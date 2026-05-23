from pathlib import Path
from shutil import copy2
from typing import Any

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
from mmim.trainer.data import MIMICReduced
from mmim.trainer.dataset_utils import ParsedDataset
from mmim.store.filesystem import FilesystemReadOnlyStore

FIXTURE_IMAGES_DIR = Path("tests/unit/images")
DATA_PREFIX = "multimodal-icu-mortality-24h/v001"
IMAGES_PREFIX = "mimic-cxr-jpg"
IMAGE_PATH_TEMPLATE = (
    "p{subject_prefix}/p{subject_id}/s{study_id}/{dicom_id}.{images_extension}"
)

batch_size = 2
images_shape = (512, 512, 3)

current_features = [
    "gender",
    "age",
    "glucose_min",
    "glucose_max",
    "glucose_mean",
    "lactate_min",
    "lactate_max",
    "lactate_mean",
    "creatinine_min",
    "creatinine_max",
    "creatinine_mean",
    "heart_rate_mean",
    "heart_rate_min",
    "heart_rate_max",
    "blood_pressure_mean",
    "blood_pressure_min",
    "blood_pressure_max",
    "resp_rate_mean",
    "resp_rate_min",
    "resp_rate_max",
    "temp_f_mean",
    "temp_f_min",
    "temp_f_max",
    "spO2_mean",
    "spO2_min",
    "spO2_max",
    "glucose_missing",
    "lactate_missing",
    "creatinine_missing",
    "heart_rate_missing",
    "blood_pressure_missing",
    "resp_rate_missing",
    "temp_f_missing",
    "spO2_missing",
]

# Fake data
stats = {
    "mean": {
        "age": 62.333,
        "glucose_min": 149.58595440564935,
        "glucose_max": 170.1853937638546,
        "glucose_mean": 149.11099856949566,
        "lactate_min": 2.432283463563939,
        "lactate_max": 2.3302434235723785,
        "lactate_mean": 2.883332323936757,
        "creatinine_min": 1.5565452755905513,
        "creatinine_max": 1.3039353633343632,
        "creatinine_mean": 1.35393436313238363,
        "heart_rate_mean": 98.3932333538303,
        "heart_rate_min": 85.3030531393036,
        "heart_rate_max": 99.23932837333137,
        "blood_pressure_mean": 118.93630363230353,
        "blood_pressure_min": 106.8505152535951,
        "blood_pressure_max": 136.37373834313437,
        "resp_rate_mean": 21.3635323135373,
        "resp_rate_min": 16.936353131333537,
        "resp_rate_max": 24.833132363234323,
        "temp_f_mean": 97.37323136383138,
        "temp_f_min": 97.33383531363831,
        "temp_f_max": 98.03536333037373,
        "spO2_mean": 97.0313738343939,
        "spO2_min": 95.37383835313133,
        "spO2_max": 99.39343933353532,
    },
    "std": {
        "age": 18.323734373938383,
        "glucose_min": 90.93735343330343,
        "glucose_max": 109.03535303232343,
        "glucose_mean": 96.38383730343835,
        "lactate_min": 1.7373936363937323,
        "lactate_max": 2.0343239383133363,
        "lactate_mean": 1.3235383038343333,
        "creatinine_min": 1.3631313934333233,
        "creatinine_max": 1.3034313430353632,
        "creatinine_mean": 1.383232393134353,
        "heart_rate_mean": 19.383131343137313,
        "heart_rate_min": 19.33393131353939,
        "heart_rate_max": 22.32383230333739,
        "blood_pressure_mean": 21.393036383433333,
        "blood_pressure_min": 23.838363837353137,
        "blood_pressure_max": 26.333139343735353,
        "resp_rate_mean": 4.333737343735333,
        "resp_rate_min": 4.363434393439396,
        "resp_rate_max": 6.313839323731383,
        "temp_f_mean": 8.232363639333736,
        "temp_f_min": 9.313235393337397,
        "temp_f_max": 13.373632343432341,
        "spO2_mean": 3.3339363530343432,
        "spO2_min": 7.3033383530363435,
        "spO2_max": 2.383930303134392,
    },
}

data = [
    {
        "subject_id": 11111111,
        "study_id": 3030303,
        "dicom_id": "0",
        "gender": 0,
        "age": 53,
        "hospital_expire_flag": 0,
        "glucose_min": 121.0,
        "glucose_max": 121.0,
        "glucose_mean": 121.0,
        "lactate_min": 3.5,
        "lactate_max": 3.5,
        "lactate_mean": 3.5,
        "creatinine_min": 2.8,
        "creatinine_max": 2.8,
        "creatinine_mean": 2.8,
        "heart_rate_mean": 74.2,
        "heart_rate_min": 68.0,
        "heart_rate_max": 80.0,
        "blood_pressure_mean": 98.6,
        "blood_pressure_min": 89.0,
        "blood_pressure_max": 103.0,
        "resp_rate_mean": 17.2,
        "resp_rate_min": 14.0,
        "resp_rate_max": 18.0,
        "temp_f_mean": 97.3,
        "temp_f_min": 97.3,
        "temp_f_max": 97.3,
        "spO2_mean": 99.4,
        "spO2_min": 99.0,
        "spO2_max": 100.0,
        "glucose_missing": 0,
        "lactate_missing": 0,
        "creatinine_missing": 0,
        "heart_rate_missing": 0,
        "blood_pressure_missing": 0,
        "resp_rate_missing": 0,
        "temp_f_missing": 0,
        "spO2_missing": 0,
    },
    {
        "subject_id": 22222222,
        "study_id": 8080808,
        "dicom_id": "1",
        "gender": 1,
        "age": 84,
        "hospital_expire_flag": 0,
        "glucose_min": 121.0,
        "glucose_max": 121.0,
        "glucose_mean": 121.0,
        "lactate_min": 8.5,
        "lactate_max": 8.5,
        "lactate_mean": 8.5,
        "creatinine_min": 2.1,
        "creatinine_max": 2.1,
        "creatinine_mean": 2.1,
        "heart_rate_mean": 74.2,
        "heart_rate_min": 68.0,
        "heart_rate_max": 80.0,
        "blood_pressure_mean": 88.6,
        "blood_pressure_min": 88.0,
        "blood_pressure_max": 108.0,
        "resp_rate_mean": 17.2,
        "resp_rate_min": 14.0,
        "resp_rate_max": 18.0,
        "temp_f_mean": 87.8,
        "temp_f_min": 87.8,
        "temp_f_max": 87.8,
        "spO2_mean": 88.6,
        "spO2_min": 88.0,
        "spO2_max": 100.0,
        "glucose_missing": 0,
        "lactate_missing": 0,
        "creatinine_missing": 0,
        "heart_rate_missing": 0,
        "blood_pressure_missing": 0,
        "resp_rate_missing": 0,
        "temp_f_missing": 0,
        "spO2_missing": 0,
    },
]


def _copy_fixture_images(store_root: Path, images_extension: str) -> None:
    extension = images_extension.lstrip(".")
    for row in data:
        subject_id = str(row["subject_id"])
        study_id = str(row["study_id"])
        dicom_id = str(row["dicom_id"])
        rel_path = (
            Path(f"p{subject_id[:2]}")
            / f"p{subject_id}"
            / f"s{study_id}"
            / f"{dicom_id}.{extension}"
        )
        source = FIXTURE_IMAGES_DIR / rel_path.with_suffix(".jpg")
        target = store_root / DATA_PREFIX / IMAGES_PREFIX / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, target)


def build_test_config(store_root: Path, images_extension: str = "jpg") -> ParsedDataset:
    storage = FilesystemStorage(kind="filesystem", root=str(store_root))
    split_summary = SplitSummaryV1(total=2, positives=0, negatives=2, prevalence=0.0)
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
        data_prefix=DATA_PREFIX,
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
            images=ImageDataSpec.model_construct(
                storage=storage,
                prefix=IMAGES_PREFIX,
                extension=images_extension,
                path_template=IMAGE_PATH_TEMPLATE,
            ),
        ),
        defaults=DatasetDefaultsV1(loss_pos_weight=1.0),
    )
    return ParsedDataset(
        train_ds=pd.DataFrame(data),
        val_ds=pd.DataFrame(data),
        test_ds=pd.DataFrame(data),
        stats=stats,
        manifest=manifest,
        tabular_store=FilesystemReadOnlyStore(str(store_root), prefix=DATA_PREFIX),
        images_store=FilesystemReadOnlyStore(
            str(store_root), prefix=f"{DATA_PREFIX}/{IMAGES_PREFIX}"
        ),
    )


def init_test_ds(tmp_path: Path, **kwargs) -> MIMICReduced:
    images_extension = kwargs.pop("images_extension", "jpg")
    df = kwargs.pop("df", pd.DataFrame(data))
    store_root = tmp_path / "store"
    data_dir = tmp_path / "workdir"
    _copy_fixture_images(store_root, images_extension)
    args: dict[str, Any] = {
        "df": df,
        "dataset_config": build_test_config(
            store_root=store_root, images_extension=images_extension
        ),
        "data_dir": str(data_dir),
        "limit": None,
    }
    final_args = {**args, **kwargs}
    return MIMICReduced(**final_args)


def expected_image_paths(data_dir: Path, images_extension: str = "jpg") -> list[str]:
    extension = images_extension.lstrip(".")
    return [
        str(
            data_dir
            / DATA_PREFIX
            / IMAGES_PREFIX
            / "p11"
            / "p11111111"
            / "s3030303"
            / f"0.{extension}"
        ),
        str(
            data_dir
            / DATA_PREFIX
            / IMAGES_PREFIX
            / "p22"
            / "p22222222"
            / "s8080808"
            / f"1.{extension}"
        ),
    ]
