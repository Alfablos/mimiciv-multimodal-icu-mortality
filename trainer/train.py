from typing import Literal

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import mlflow
import mlflow.pytorch

import torch.cuda
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .data import MIMICReduced
from .gradcam import grad_cam
from .models.fusion import Fusion
import mlflow
import os
import json
from argparse import Namespace

import pandas as pd
import torch
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
import mlflow.pytorch


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
from .meta import log_metadata
from .config import model_selection_metric


def is_better_score(
    current: float, best: float | None, mode: Literal["lower", "higher"]
):
    if best is None:
        return True

    if mode == "higher":
        return current > best

    return current < best


def log_model(model: Fusion, epoch: int, metrics: dict[str, float]):
    mlflow.pytorch.log_model(model, name="multimodal_icu_mortality")
    mlflow.log_metric("best_epoch", epoch)
    mlflow.log_metric("best_val_loss", metrics["val_loss"])
    mlflow.log_metric("best_val_auroc", metrics["AUROC"])
    mlflow.log_metric("best_val_auprc", metrics["AUPRC"])
    mlflow.log_metric("best_val_sensitivity_at_95_spec", metrics["sens_at_95_spec"])
    mlflow.set_tag("best_model.logged", "true")
    mlflow.set_tag("best_model.epoch", str(epoch))
    mlflow.set_tag("best_model.selection_metric", model_selection_metric)


def upload_gradcam(
    images: Tensor,
    tabs: Tensor,
    model: Fusion,
    epoch_n: int,
    purpose: Literal["train", "val"],
):
    model_was_training = model.training
    model.eval()
    try:  # if anything fails the model is back to training mode
        for i in range(min(3, images.size(0))):
            image_t = images[i : i + 1]  # so it stays a 4D tensor
            tab_t = tabs[i : i + 1]
            fig = grad_cam(
                model=model,
                image_tensor=image_t,
                tab_tensor=tab_t,
                transform_images=False,  # they've already transformed by the trining loop!
            )
            mlflow.log_figure(
                figure=fig,
                artifact_file=f"gradcam/epoch_{epoch_n:03d}/{purpose}_{i}.png",
            )
    finally:
        if model_was_training:
            model.train()


def train(
    model: Fusion,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    best_metric: float | None = None

    try:
        for epoch in range(epochs):
            model.train()
            print(f"Starting epoch {epoch}.")

            losses = []

            for batch_n, (images, tabs, labels) in enumerate(train_loader):
                images, tabs, labels = (
                    images.to(device),
                    tabs.to(device),
                    labels.unsqueeze(dim=1).to(device),
                )
                images = MIMICReduced.gpu_transforms(images)

                optimizer.zero_grad()
                # preds is called like that to provide a uniform interface
                # but the model is applying no activation function, actually ouputting logits.
                # the curremt loss function is BCEWithLogitsLoss
                # Careful when changing the loss function
                preds: Tensor = model(images, tabs)
                loss: Tensor = loss_fn(preds, labels)
                loss.backward()
                losses.append(float(loss.item()))
                print(
                    f"Train epoch {epoch} batch {batch_n} of {len(train_loader)} | loss:",
                    loss.item(),
                )
                optimizer.step()  # must happen before to avoid zeroing the gradients

            mean_loss = float(np.mean(losses))
            print("Sending training metrics and artifacts to mlflow")

            mlflow.log_metric("train_loss", mean_loss, step=epoch)

            upload_gradcam(
                images=images,
                tabs=tabs,
                model=model,
                epoch_n=epoch,
                purpose="train",
            )

            # Only doing this on the validation set, the primary overfitting indicator
            # is the raw loss.
            metrics = evaluate(
                model=model,
                val_loader=val_loader,
                device=device,
                loss_fn=loss_fn,
                epoch_n=epoch,
            )

            current_score = float(metrics[model_selection_metric])
            if is_better_score(current_score, best_metric, mode="higher"):
                best_metric = current_score
                log_model(model=model, epoch=epoch, metrics=metrics)

        mlflow.set_tag("training.status", "completed")
        print("Training done.")

    except KeyboardInterrupt:
        print("User interrupted the training job.")
        mlflow.set_tag("training.status", "interrupted")
        print("Exiting.")


def evaluate(
    model: Fusion,
    val_loader: DataLoader,
    device: Literal["cuda", "cpu"],
    loss_fn: nn.Module,
    epoch_n: int,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        preds = []
        labels = []

        val_losses = []

        for batch_n, (val_images, val_tabs, val_labels) in enumerate(val_loader):
            val_images, val_tabs, val_labels = (
                val_images.to(device),
                val_tabs.to(device),
                val_labels.unsqueeze(dim=1).to(device),
            )
            val_images: Tensor = MIMICReduced.gpu_transforms(val_images)

            val_preds: Tensor = model(val_images, val_tabs)
            val_loss: Tensor = loss_fn(val_preds, val_labels)
            print(
                f"Validation epoch {epoch_n} batch {batch_n} of {len(val_loader)} | loss:",
                val_loss.item(),
            )
            val_losses.append(val_loss.item())

            # logits => probabilities
            pred_probs = torch.sigmoid(val_preds).cpu()
            preds.append(pred_probs)

            labels.append(val_labels.cpu())

        # preds and labels are lists of lists
        preds = torch.cat(preds).numpy()  # now flat
        labels = torch.cat(labels).numpy()

        val_mean_loss = np.mean(val_losses)
        mlflow.log_metric("val_loss", val_mean_loss, step=epoch_n)

        metrics = get_metrics(preds, labels)
        metrics["val_loss"] = val_mean_loss
        auroc = metrics["AUROC"]
        auprc = metrics["AUPRC"]
        sens_at_95_spec = metrics["sens_at_95_spec"]
        mlflow.log_metric("val_auroc", auroc, step=epoch_n)
        mlflow.log_metric("val_auprc", auprc, step=epoch_n)
        mlflow.log_metric("val_sens_at_95_spec", sens_at_95_spec, step=epoch_n)
        print(
            f"Epoch {epoch_n} (VAL):\n"
            f"AUROC: {metrics['AUROC']}\n"
            f"AUPRC: {metrics['AUPRC']}\n"
            f"Sensitivity at 95% specificity: {metrics['sens_at_95_spec']}\n"
        )
    upload_gradcam(
        images=val_images, tabs=val_tabs, model=model, epoch_n=epoch_n, purpose="val"
    )
    return metrics


def get_metrics(preds, labels) -> dict[str, float]:
    auroc = float(roc_auc_score(y_true=labels, y_score=preds))
    auprc = float(average_precision_score(y_true=labels, y_score=preds))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true=labels, y_score=preds
    )
    # Specificity = 1 - false_positive_rate
    # Selecting false_positive_rate below 0.05!
    under_005_indices = np.where(false_positive_rate <= 0.05)[0]
    sensitivity_at_95_perc_spec = (
        true_positive_rate[under_005_indices[-1]] if len(under_005_indices) > 0 else 0.0
    )
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "sens_at_95_spec": sensitivity_at_95_perc_spec,
    }


def train_start(_: Namespace):
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
        + "_epochs"
        + str(hyperparameters["epochs"])
        + "_dropout"
        + str(hyperparameters["dropout"])
        + (
            "_trainlimit" + str(hyperparameters["train_limit"])
            if hyperparameters["train_limit"] != 1.0
            else ""
        )
    ) as run:
        metadata = log_metadata()
        for k, v in metadata.items():
            print(f"{k} => {v}")
        mlflow.log_params(
            {f"hyperparameters.{k}": v for k, v in hyperparameters.items()}
        )
        mlflow.set_tag("mlflow.run_id", run.info.run_id)  # for easy retrieval later

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
