import pandas as pd
# import pylibjpeg # decompression backend for dicom pixel data #TODO


import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transformsV2
import torchvision.io as tvio

from .models import vision_encoder
from .transforms import PadToSquare


class MIMICReduced(Dataset):
    gpu_transforms = transformsV2.Compose(
        [
            # --- moved from CPU to speed things up
            transformsV2.ToImage(),
            # ---
            transformsV2.ToDtype(torch.float32, scale=True),
            transformsV2.Normalize(
                mean=vision_encoder.DEFAULT_WEIGHTS.transforms().mean,
                std=vision_encoder.DEFAULT_WEIGHTS.transforms().std,
            ),
        ]
    )

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_stats: dict[str, dict[str, float]],
        images_base_dir: str,
        images_extension: str = "dcm",
        label_column: str = "hospital_expire_flag",
        debug: bool = False,
        limit: float | None = None,
        cpu_transforms=transformsV2.Compose(
            [
                PadToSquare(),
                transformsV2.Resize((512, 512), antialias=True),  # cannot resize on GPU
                # transformsV2.ToImage()
            ]
        ),
    ):
        super().__init__()
        if images_extension.lower().lstrip(".") not in ["jpg", "dcm", "dicom"]:
            raise ValueError(f"Extension {images_extension} is not supported.")

        if limit and not 0.0 < limit <= 1.0:
            raise ValueError("Invalid value for limit:", limit)
        elif limit:
            df = df.sample(frac=limit, random_state=42).reset_index(drop=True)

        self.debug = debug
        self.transforms = cpu_transforms
        self.y: Tensor = torch.tensor(df[label_column].values, dtype=torch.float32)
        self.images_extension = images_extension.rstrip("/").lower()

        subject_ids = df["subject_id"].astype(str)
        study_ids = df["study_id"].astype(str)
        dicom_ids = df["dicom_id"].astype(str)
        df["image_path"] = (
            images_base_dir.rstrip("/")
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
        df = df.drop(["subject_id", "study_id", "dicom_id"], axis=1)

        self.image_paths = df["image_path"].values

        if debug:
            df.to_csv("mimicreduced_debug.csv")

        features_df = df.drop([label_column, "image_path"], axis=1)
        self.X: Tensor = torch.tensor(features_df.values, dtype=torch.float32)
        self.features: list[str] = features_df.columns.tolist()

        continuous_features = [
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
        ]

        # mean 0 and std 1 (neutral) for non-continuous features
        self.mean = torch.tensor(
            [
                dataset_stats["mean"][col] if col in continuous_features else 0
                for col in self.features
            ],
            dtype=torch.float32,
        )  # self.features guaranties the order and fails if misaligned
        self.std = torch.tensor(
            [
                dataset_stats["std"][col] if col in continuous_features else 1
                for col in self.features
            ],
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i) -> tuple[Tensor, Tensor, Tensor]:
        image_path = self.image_paths[i]

        if self.images_extension == "dcm" or self.images_extension == "dicom":
            # 1. Decompress ACCORDING TO PYDICOM DOCS? pylibjpeg is available
            # 2. Apply VOI LUTs (Value of Interest Look-Up Tables) to standardize pixel values
            #    across vendors #TODO

            # # minmax normalization to reduce peaks of 12-16 bit dicom image
            # pixels = (pixels - min(pixels)) / (max(pixels) - min(pixels) + 1e-8)
            # print(data.PhotochromaticInterpretation)
            # print(pixels.shape)
            raise NotImplementedError("Not yet implemented")

        else:  # jpg
            image = tvio.read_image(image_path, mode=tvio.ImageReadMode.RGB)

        # check if the image needs padding to have a 1:1 ration before resize
        image = self.transforms(image)
        x = self.normalize(self.X[i])
        y = self.y[i]
        return image, x, y

    def normalize(self, t: Tensor) -> Tensor:
        return (t - self.mean) / (self.std + 1e-8)

    def stats(self):
        return (self.mean, self.std)


if __name__ == "__main__":
    # with open('./dataset/stats.json', 'r') as s:
    #     ds = MIMICReduced(
    #         df=pd.read_csv("./dataset/ds_train.csv"),
    #         label_column="hospital_expire_flag",
    #         images_extension="jpg",
    #         images_base_dir="../mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.1.0/files",
    #         dataset_stats=json.load(s)
    #     )
    pass
