from mmim.generator.manifest import ManifestV1
from mmim.store.filesystem import FilesystemReadOnlyStore
from mmim.trainer.dataset_utils import (
    ParsedDataset,
    manifest_from_uri,
    parsed_dataset_from_manifest,
)
from typing import Literal, Generator, Any
from pydantic import BaseModel
from datetime import datetime, UTC

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import mlflow
import mlflow.pytorch

import torch.cuda
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mlflow
from argparse import Namespace

import torch
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
import mlflow.pytorch


from data import MIMICReduced
from gradcam import grad_cam
from models.fusion import Fusion
from config import dataset_shuffle, num_workers, debug, Hyperparameters
from config import model_selection_metric


class Metrics(BaseModel):
    AUROC: float
    AUPRC: float
    sens_at_95_spec: float


# import-time validation since the value of model_selection_metric is already known at that point
# pydantic would only validate it on first instatiation
VALID_MODEL_SELECTION_METRICS = set(Metrics.model_fields)

if model_selection_metric not in VALID_MODEL_SELECTION_METRICS:
    raise ValueError(
        f"Invalid model_selection_metric={model_selection_metric}. "
        f"Expected one of: {', '.join(sorted(VALID_MODEL_SELECTION_METRICS))}"
    )

# completed: completed, interrupted: user interrupt, failed: handled fatal exceptions
type TrainStatus = Literal["completed", "interrupted", "failed"]


class LoggedModel(BaseModel):
    model: Any
    epoch: int
    metrics: dict[str, int | float]
    metadata: dict[str, int | float | str]
    train_loss: float
    val_loss: float
    train_start_time: str
    train_end_time: str
    selection_metric: str
    selection_metric_value: float


class TrainLoopResult(BaseModel):
    start_time: str
    end_time: str
    train_status: TrainStatus


class TrainingResult(BaseModel):
    dataset_manifest: ManifestV1
    train_results: TrainLoopResult
    run_id: str


def is_better_score(
    current: float, best: float | None, mode: Literal["lower", "higher"]
):
    if best is None:
        return True

    if mode == "higher":
        return current > best

    return current < best


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
    verbose: bool = False,
) -> Generator[LoggedModel, None, TrainLoopResult]:
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    assert device in ["cuda", "cpu"]

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    start_time_str = datetime.now(UTC).strftime("%Y%m%d_%H%M")
    try:
        for epoch in range(epochs):
            model.train()
            if verbose:
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
                # the current loss function is BCEWithLogitsLoss
                # Careful when changing the loss function
                preds: Tensor = model(images, tabs)
                loss: Tensor = loss_fn(preds, labels)
                loss.backward()
                losses.append(float(loss.item()))

                if verbose:
                    print(
                        f"Train epoch {epoch} batch {batch_n} of {len(train_loader)} | loss:",
                        loss.item(),
                    )

                optimizer.step()  # must happen before to avoid zeroing the gradients
            end_time_str = datetime.now(UTC).strftime("%Y%m%d_%H%M")
            mean_loss = float(np.mean(losses))

            # print("Sending training metrics and artifacts to mlflow")
            # mlflow.log_metric("train_loss", mean_loss, step=epoch)
            #
            # upload_gradcam(
            #     images=images,
            #     tabs=tabs,
            #     model=model,
            #     epoch_n=epoch,
            #     purpose="train",
            # )

            # Only doing this on the validation set, the primary overfitting indicator
            # is the raw loss.
            metrics, val_loss = evaluate(
                model=model,
                val_loader=val_loader,
                device=device,
                loss_fn=loss_fn,
                epoch_n=epoch,
                verbose=verbose,
            )

            float_val_loss = float(val_loss)  # converts np.float64
            # model_name = f"{experiment_family}@{start_time_str}_e{epoch}"
            best_model_selection_metric_value = getattr(metrics, model_selection_metric)
            model_metadata: dict[str, str | int | float] = {
                "epoch": epoch,
                "loss": float_val_loss,
                "auroc": metrics.AUROC,
                "auprc": metrics.AUPRC,
                "sens_at_95_spec": metrics.sens_at_95_spec,
                "time": datetime.now(UTC).strftime("%Y%m%d_%H%M"),
                "selection_metric": model_selection_metric,
                "selection_metric_value": best_model_selection_metric_value,
            }
            yield LoggedModel(
                model=model,
                epoch=epoch,
                metrics=metrics.model_dump(),
                metadata=model_metadata,
                val_loss=float_val_loss,
                train_loss=mean_loss,
                train_start_time=start_time_str,
                train_end_time=end_time_str,
                selection_metric=model_selection_metric,
                selection_metric_value=best_model_selection_metric_value,
            )

        train_status = "completed"
    except KeyboardInterrupt:
        print("User interrupted the training job.")
        mlflow.set_tag("training.status", "interrupted")
        print("Exiting.")
        train_status = "interrupted"

    mlflow.set_tag("training.status", "completed")
    if verbose:
        print("Training done.")

    end_time_str = datetime.now(UTC).strftime("%Y%m%d_%H%M")

    assert train_status in ["interrupted", "completed", "failed"]

    return TrainLoopResult(
        start_time=start_time_str,
        end_time=end_time_str,
        train_status=train_status,
    )


def evaluate(
    model: Fusion,
    val_loader: DataLoader,
    device: Literal["cuda", "cpu"],
    loss_fn: nn.Module,
    epoch_n: int,
    verbose: bool = False,
) -> tuple[Metrics, float]:
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
            if verbose:
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
        # mlflow.log_metric("val_loss", val_mean_loss, step=epoch_n)

        metrics = get_metrics(preds, labels)
        val_auroc: float = metrics.AUROC
        val_auprc: float = metrics.AUPRC
        val_sens_at_95_spec: float = metrics.sens_at_95_spec
        # mlflow.log_metric("val_auroc", val_auroc, step=epoch_n)
        # mlflow.log_metric("val_auprc", val_auprc, step=epoch_n)
        # mlflow.log_metric("val_sens_at_95_spec", val_sens_at_95_spec, step=epoch_n)
        if verbose:
            print(
                f"Epoch {epoch_n} (VAL):\n"
                f"AUROC: {val_auroc}\n"
                f"AUPRC: {val_auprc}\n"
                f"Sensitivity at 95% specificity: {val_sens_at_95_spec}\n"
            )
    # upload_gradcam(
    #     images=val_images, tabs=val_tabs, model=model, epoch_n=epoch_n, purpose="val"
    # )
    return metrics, val_mean_loss


def get_metrics(preds, labels) -> Metrics:
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
    return Metrics(
        AUROC=auroc,
        AUPRC=auprc,
        sens_at_95_spec=sensitivity_at_95_perc_spec,
    )


def train_cli(args: Namespace):
    manifest = manifest_from_uri(args.manifest_uri)
    ds_config = parsed_dataset_from_manifest(manifest)

    train_result = start_train(
        dataset_config=ds_config,
        hyperparameters=Hyperparameters(),
        working_directory=args.working_directory,
        verbose=args.verbose,
    )

    print(train_result)


def start_train(
    dataset_config: ParsedDataset,
    hyperparameters: Hyperparameters,
    working_directory: str = "./out",
    verbose: bool = False,
) -> TrainingResult:
    if hyperparameters.train_limit != 1.0:
        print(
            f"WARNING: train_limit is set to {hyperparameters.train_limit}, make sure loss_pos_weight is still valid."
        )

    working_directory = (
        f"{dataset_config.tabular_store.dir}"
        if isinstance(dataset_config.tabular_store, FilesystemReadOnlyStore)
        else working_directory
    )

    train_ds = MIMICReduced(
        df=dataset_config.train_ds,
        dataset_config=dataset_config,
        data_dir=working_directory,
        debug=debug,
        limit=hyperparameters.train_limit,
    )

    train_dl = DataLoader(
        pin_memory=torch.cuda.is_available(),
        dataset=train_ds,
        shuffle=dataset_shuffle,
        batch_size=hyperparameters.batch_size,
        num_workers=num_workers,
    )

    val_ds = MIMICReduced(
        df=dataset_config.val_ds,
        dataset_config=dataset_config,
        data_dir=working_directory,
        debug=debug,
        limit=hyperparameters.train_limit,
    )

    if train_ds.features != val_ds.features:
        raise ValueError(
            f"Train dataset features differ from validation dataset.\nTrain: {','.join(train_ds.features)}\nValidation: {','.join(val_ds.features)}"
        )

    model = Fusion(
        dropout=hyperparameters.dropout, tab_features_in=len(train_ds.features)
    )

    val_dl = DataLoader(
        pin_memory=torch.cuda.is_available(),
        dataset=val_ds,
        shuffle=False,  # making val_ds more deterministic
        batch_size=hyperparameters.batch_size,
        num_workers=num_workers,
    )

    loss_fn = BCEWithLogitsLoss(
        # this tensor is still on the CPU
        # be sure to move it to(device)
        pos_weight=torch.tensor(
            [dataset_config.manifest.defaults.loss_pos_weight]
        )  # so pytorch is free to broadcast it
    )

    optimizer = AdamW(model.parameters(), lr=hyperparameters.learning_rate)

    train_gen = train(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=hyperparameters.epochs,
        train_loader=train_dl,
        val_loader=val_dl,
        verbose=verbose,
    )

    while True:
        try:
            m = next(train_gen)
            print("Current train loss:" + str(m.train_loss))

        except StopIteration as e:
            return e.value
