from os import cpu_count

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import DenseNet
from torch.nn import BCEWithLogitsLoss
import mlflow
import mlflow.pytorch

from models.vision_encoder import Xencoder
from data import MIMICReduced
from config import (
    loss_pos_weight,
    dataset_shuffle,
    num_workers,
    hyperparameters
)
from models.fusion import Fusion
from train import train


if __name__ == '__main__':
    mlflow.set_experiment('multimodal ICU mortality')
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)
    with mlflow.start_run():
        mlflow.log_params(hyperparameters)
        model = Fusion(
            dropout=hyperparameters['dropout']
        )

        train_ds = MIMICReduced(
            df=pd.read_csv('./ds_train.csv'),
            label_column='hospital_expire_flag',
            images_extension='jpg',
            images_base_dir='../mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.1.0/files',
            debug=True,
            limit=hyperparameters['train_limit']
        )
        train_dl = DataLoader(
            pin_memory=True,
            dataset=train_ds,
            shuffle=dataset_shuffle,
            batch_size=hyperparameters['batch_size'],
            num_workers=num_workers
        )

        val_ds = MIMICReduced(
            df=pd.read_csv('./ds_val.csv'),
            label_column='hospital_expire_flag',
            images_extension='jpg',
            images_base_dir='../mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.1.0/files',
            debug=True,
            # optional for the validation set as well but
            # allowes me to iterate faster
            limit=hyperparameters['train_limit']
        )
        val_dl = DataLoader(
            pin_memory=True,
            dataset=val_ds,
            shuffle=dataset_shuffle,
            batch_size=hyperparameters['batch_size'],
            num_workers=num_workers
        )

        loss_fn = BCEWithLogitsLoss(
            # this tensor is still on the CPU
            # be sure to move it to(device)
            pos_weight=torch.tensor([loss_pos_weight]) # so pytorch is free to broadcast it
        )

        optimizer = AdamW(
            model.parameters(),
            lr=hyperparameters['learning_rate']
        )

        train(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=hyperparameters['epochs'],
            train_loader=train_dl,
            val_loader=val_dl
        )
