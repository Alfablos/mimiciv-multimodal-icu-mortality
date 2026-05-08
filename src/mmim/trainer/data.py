from pathlib import Path
from mmim.trainer.dataset_utils import ParsedDataset
import pandas as pd


import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transformsV2
import torchvision.io as tvio

from .models import vision_encoder
from .transforms import PadToSquare
from .utils import ImageResolver

supported_image_extensions = ["jpg", "jpeg", "dcm", "dicom"]


extended_image_extensions = supported_image_extensions + [
    f".{e}" for e in supported_image_extensions
]


class MMIMICImageResolver(ImageResolver):
    def __init__(
        self,
        base_dir: str,
        images_prefix: str,
        path_template: str,
        images_extension: str,
    ):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.images_prefix = images_prefix
        self.path_template = path_template
        self.images_extension = images_extension

    def resolve(self, row: pd.Series) -> str:
        subject_id = str(row["subject_id"])
        study_id = str(row["study_id"])
        dicom_id = str(row["dicom_id"])

        rel_path = self.path_template.format(
            subject_prefix=subject_id[:2],
            subject_id=subject_id,
            study_id=study_id,
            dicom_id=dicom_id,
            images_extension=self.images_extension,
        )

        return str(self.base_dir / self.images_prefix / rel_path)


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
        dataset_config: ParsedDataset,
        data_dir: str,
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

        dataset_stats = dataset_config.stats
        label_column = dataset_config.manifest["data"]["tabular"]["label_column"]
        images_extension = dataset_config.manifest["data"]["images"]["extension"]

        if images_extension.lower().lstrip(".") not in extended_image_extensions:
            raise ValueError(
                f"Extension {images_extension} is not supported. Supported extensions are: {', '.join(supported_image_extensions)}"
            )

        if limit and not 0.0 < limit <= 1.0:
            raise ValueError("Invalid value for limit:", limit)
        elif limit:
            df = df.sample(frac=limit, random_state=42).reset_index(drop=True)

        self.debug = debug
        self.transforms = cpu_transforms
        self.y: Tensor = torch.tensor(df[label_column].values, dtype=torch.float32)
        self.images_extension = images_extension

        images_prefix = Path(dataset_config.data_prefix) / Path(
            dataset_config.images_prefix
        )
        image_resolver = MMIMICImageResolver(
            base_dir=data_dir,
            images_prefix=str(images_prefix),
            path_template=dataset_config.images_path_template,
            images_extension=images_extension,
        )
        df["image_path"] = df.apply(
            image_resolver.resolve, axis=1
        )  # pass rows, not columns

        print("Downloading missing images...")
        store = dataset_config.store

        for local_path in df["image_path"]:
            l_path = Path(local_path)
            rel_path = str(
                l_path.relative_to(Path(data_dir) / images_prefix)
            )  # the store doesn't know about the datadir, we changed the prefix
            store_key = str(images_prefix / rel_path)

            if not l_path.exists():
                l_path.parent.mkdir(exist_ok=True, parents=True)
                print(f"Reading from store {store_key} and copying it to {l_path}...")
                bytes = store.read_bytes(
                    store_key, with_prefix=False
                )  # manually handling the prefix

                print(f"Writing to {l_path}")
                with open(str(l_path), "wb") as f:
                    f.write(bytes)
            else:
                print(f"{l_path} already exists.")
        print("Done.")

        self.image_paths = df["image_path"].values

        df = df.drop(["subject_id", "study_id", "dicom_id"], axis=1)

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
