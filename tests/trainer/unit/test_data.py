from pytest import raises
import torch
from typing import Any
import pandas as pd

from trainer.data import MIMICReduced

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


def init_test_ds(**kwargs) -> MIMICReduced:
    args: dict[str, Any] = {
        "df": pd.DataFrame(data),
        "dataset_stats": stats,
        "label_column": "hospital_expire_flag",
        "images_extension": "jpg",
        "images_base_dir": "./tests/trainer/unit/images",
        "limit": None,
    }
    final_args = {**args, **kwargs}
    return MIMICReduced(**final_args)


def test_ds_has_right_features():
    test_ds = init_test_ds()
    assert test_ds.features == current_features


def test_ds_allowed_image_extensions_are_ok():
    for allowed_ext in ["jpg", ".jpg", "dcm", ".dcm", "dicom", ".dicom"]:
        _ = init_test_ds(images_extension=allowed_ext)


def test_ds_wrong_image_extensions_are_rejected():
    with raises(ValueError, match="Extension .+ is not supported."):
        init_test_ds(images_extension="unallowed")
        init_test_ds(images_extension=".unallowed")
        init_test_ds(images_extension="#? unallowed")


def test_ds_returns_images_correctly():
    ds = init_test_ds()
    paths = [
        "./tests/trainer/unit/images/p11/p11111111/s3030303/0.jpg",
        "./tests/trainer/unit/images/p22/p22222222/s8080808/1.jpg",
    ]

    for i, path in enumerate(paths):
        img, example, label = ds[i]
        assert example.shape.numel() == 34, "Wrong shape for training example"
        assert label.shape.numel() == 1, "Wrong shape for label"
        assert img.shape == torch.Size([3, 512, 512]), (
            f"Wrong shape for image: {img.shape}. Should be [3, 512, 512]"
        )
        assert ds.image_paths[i] == path
