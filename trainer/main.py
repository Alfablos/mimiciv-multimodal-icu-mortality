import mlflow
import os
import json

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
import mlflow.pytorch


from .data import MIMICReduced
from .config import (
    loss_pos_weight,
    dataset_stats_file,
    dataset_shuffle,
    num_workers,
    hyperparameters,
    image_base_dir,
    image_extension,
    train_csv,
    val_csv,
    debug,
)
from .models.fusion import Fusion
from .train import train
from .meta import log_metadata


if __name__ == "__main__":
    if hyperparameters["train_limit"] != 1.0:
        print(
            f"WARNING: train_limit is set to {hyperparameters['train_limit']}, make sure loss_pos_weight is still valid."
        )
    with open(dataset_stats_file, "r") as f:
        ds_stats = json.load(f)

    mlflow.set_experiment(
        os.getenv("MLFLOW_EXPERIMENT_NAME", "Multimodal ICU mortality")
    )
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(5)

    with mlflow.start_run(
        run_name="fusion_bs"
        + str(hyperparameters["batch_size"])
        + "_lr"
        + str(hyperparameters["learning_rate"])
        + "_epocs"
        + str(hyperparameters["epochs"])
        + "_dropout"
        + str(hyperparameters["dropout"])
        + (
            "_trainlimit" + str(hyperparameters["train_limit"])
            if hyperparameters["train_limit"] != 1.0
            else ""
        )
    ):
        metadata = log_metadata()
        for k, v in metadata.items():
            print(f"{k} => {v}")
        mlflow.log_params(
            {f"hyperparameters.{k}": v for k, v in hyperparameters.items()}
        )

        model = Fusion(dropout=hyperparameters["dropout"])

        train_ds = MIMICReduced(
            df=pd.read_csv(train_csv),
            dataset_stats=ds_stats,
            label_column="hospital_expire_flag",
            images_extension=image_extension,
            images_base_dir=image_base_dir,
            debug=debug,
            limit=hyperparameters["train_limit"],
        )
        train_dl = DataLoader(
            pin_memory=True,
            dataset=train_ds,
            shuffle=dataset_shuffle,
            batch_size=hyperparameters["batch_size"],
            num_workers=num_workers,
        )

        val_ds = MIMICReduced(
            df=pd.read_csv(val_csv),
            dataset_stats=ds_stats,
            label_column="hospital_expire_flag",
            images_extension=image_extension,
            images_base_dir=image_base_dir,
            debug=debug,
            # optional for the validation set as well but
            # allowes me to iterate faster
            limit=hyperparameters["train_limit"],
        )

        val_dl = DataLoader(
            pin_memory=True,
            dataset=val_ds,
            shuffle=dataset_shuffle,
            batch_size=hyperparameters["batch_size"],
            num_workers=num_workers,
        )

        loss_fn = BCEWithLogitsLoss(
            # this tensor is still on the CPU
            # be sure to move it to(device)
            pos_weight=torch.tensor(
                [loss_pos_weight]
            )  # so pytorch is free to broadcast it
        )

        optimizer = AdamW(model.parameters(), lr=hyperparameters["learning_rate"])

        train(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=hyperparameters["epochs"],
            train_loader=train_dl,
            val_loader=val_dl,
        )
