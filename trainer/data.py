import numpy as np
import pandas as pd
import pydicom
import pylibjpeg # decompression backend for dicom pixel data #TODO

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transformsV2
import torchvision.io as tvio

from models import vision_encoder
import image


class MIMICReduced(Dataset):
    gpu_transforms = transformsV2.Compose([
        # --- moved from CPU to speed things up
        transformsV2.ToImage(),
        # ---
        transformsV2.ToDtype(torch.float32, scale=True),
        transformsV2.Normalize(
            mean=vision_encoder.DEFAULT_WEIGHTS.transforms().mean,
            std=vision_encoder.DEFAULT_WEIGHTS.transforms().std
        )])

    def __init__(
            self,
            df: pd.DataFrame,
            dataset_stats: dict,
            images_base_dir: str,
            images_extension: str = 'dcm',
            label_column: str = 'hospital_expire_flag',
            debug: bool = False,
            limit: float | None = None,
            cpu_transforms = transformsV2.Compose([
                image.PadToSquare(),
                transformsV2.Resize((512, 512), antialias=True), # cannot resize on GPU
                # transformsV2.ToImage()
            ])
    ):
        super().__init__()
        if not images_extension.lower().lstrip('.') in ['jpg', 'dcm', 'dicom']:
            raise ValueError(f'Extension {images_extension} is not supported.')

        if limit and not 0.0 < limit <= 1.0:
            raise ValueError('Invalid value for limit:', limit)
        elif limit:
            df = df.sample(frac=limit, random_state=42).reset_index(drop=True)

        self.debug = debug
        self.transforms = cpu_transforms
        self.y: Tensor = torch.tensor(df[label_column].values, dtype=torch.float32)
        self.images_extension = images_extension.rstrip('/').lower()

        subject_ids = df['subject_id'].astype(str)
        study_ids = df['study_id'].astype(str)
        dicom_ids = df['dicom_id'].astype(str)
        df['image_path'] = images_base_dir.rstrip('/') + '/p' + subject_ids.str[:2] + '/p' + subject_ids + '/s' + study_ids + '/' + dicom_ids + '.' + images_extension.lstrip('.')
        df = df.drop(['subject_id', 'study_id', 'dicom_id'], axis=1)

        self.image_paths = df['image_path'].values


        if debug:
            df.to_csv('mimicreduced_debug.csv')

        features_df = df.drop([label_column, 'image_path'], axis=1)
        self.X: Tensor = torch.tensor(features_df.values, dtype=torch.float32)
        self.features: list[str] = features_df.columns.tolist()

        self.mean = torch.tensor(dataset_stats['mean'][col] for col in dataset_stats['mean'].keys())
        self.std = torch.tensor(dataset_stats['std'][col] for col in dataset_stats['std'].keys())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i) -> tuple[Tensor, Tensor, Tensor]:
        image_path = self.image_paths[i]

        if self.images_extension == 'dcm' or self.images_extension == 'dicom':
            data = pydicom.dcmread(image_path)
            pixels = data.pixel_array
            print('', data.file_meta.TransferSyntaxUID)
            
            # 1. Decompress ACCORDING TO PYDICOM DOCS? pylibjpeg is available
            # 2. Apply VOI LUTs (Value of Interest Look-Up Tables) to standardize pixel values
            #    across vendors #TODO
            
            # # minmax normalization to reduce peaks of 12-16 bit dicom image
            # pixels = (pixels - min(pixels)) / (max(pixels) - min(pixels) + 1e-8)
            # print(data.PhotochromaticInterpretation)
            # print(pixels.shape)
            image = torch.tensor([], dtype=torch.float32)
            raise NotImplementedError('Not yet implemented')

        else: # jpg
            image = tvio.read_image(image_path, mode=tvio.ImageReadMode.RGB)


        # check if the image needs padding to have a 1:1 ration before resize
        image = self.transforms(image)
        x = (self.X[i] / self.mean) / (self.std + 1e-8)
        y = self.y[i]
        return image, x, y


if __name__ == '__main__':
    train_ds = MIMICReduced(
        df=pd.read_csv('./ds_train.csv'),
        label_column='hospital_expire_flag',
        images_extension='jpg',
        images_base_dir='../mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.1.0/files'
    )

    img, example, label = train_ds[0]
    tvio.write_jpeg(img, 'debug.jpg')
    assert example.shape.numel() == 34, 'Wrong shape for training example'
    assert label.shape.numel() == 1, 'Wrong shape for label'
