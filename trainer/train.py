from typing import Literal

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import mlflow
import mlflow.pytorch

import torch.cuda
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from data import MIMICReduced


def train(
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader
):
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    for epoch in range(epochs):
        model.train()
        print(f'Starting epoch {epoch}.')

        for batch_n, (images, tabs, labels) in enumerate(train_loader):
            images, tabs, labels = images.to(device), tabs.to(device), labels.unsqueeze(dim=1).to(device)
            images = MIMICReduced.gpu_transforms(images)

            optimizer.zero_grad()
            preds = model(images, tabs)
            loss = loss_fn(preds, labels)
            loss.backward()

            if batch_n == len(train_loader) - 1:
                print('Sending training loss to mlflow')
                mlflow.log_metric('train_loss', loss.item(), step=epoch)

            print(f'Train epoch {epoch} batch {batch_n} of {len(train_loader)} | loss:', loss.item())
            optimizer.step()

        # Only doing this on the validation set, the primary overfitting indicator
        # is the raw loss.
        evaluate(
            model=model,
            val_loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            epoch_n=epoch
        )
        mlflow.pytorch.log_model(model, name='multimodal_icu_mortality')
    print('Training done.')


def evaluate(
        model: nn.Module,
        val_loader: DataLoader,
        device: Literal['cuda', 'cpu'],
        loss_fn: nn.Module,
        epoch_n: int
):
    model.eval()
    with torch.no_grad():
        preds = []
        labels = []

        for batch_n, (val_images, val_tabs, val_labels) in enumerate(val_loader):
            val_images, val_tabs, val_labels = val_images.to(device), val_tabs.to(device), val_labels.unsqueeze(dim=1).to(device)
            val_images = MIMICReduced.gpu_transforms(val_images)

            val_preds = model(val_images, val_tabs)
            val_loss = loss_fn(val_preds, val_labels)
            print(f'Validation epoch {epoch_n} batch {batch_n} of {len(val_loader)} | loss:', val_loss.item())

            # logits => probabilities
            pred_probs = torch.sigmoid(val_preds).cpu()
            preds.append(pred_probs)

            labels.append(val_labels.cpu())
            if batch_n == len(val_loader) - 1:
                mlflow.log_metric('val_loss', val_loss.item(), step=epoch_n)

        # preds and labels are lists of lists
        preds = torch.cat(preds).numpy()    # now flat
        labels = torch.cat(labels).numpy()

        metrics = calculate_metrics(preds, labels)
        auroc = metrics['AUROC']
        auprc = metrics['AUPRC']
        sens_at_95_spec = metrics['sens_at_95_spec']
        mlflow.log_metric('val_auroc', auroc, step=epoch_n)
        mlflow.log_metric('val_auprc', auprc, step=epoch_n)
        mlflow.log_metric('val_sens_at_95_spec', sens_at_95_spec, step=epoch_n)
        print(f'Epoch {epoch_n} (VAL):\nAUROC: {metrics['AUROC']}\nAUPRC: {metrics['AUPRC']}\nSensitivity at 95% specificity: {metrics['sens_at_95_spec']}\n')




def calculate_metrics(preds, labels):
    auroc = roc_auc_score(y_true=labels, y_score=preds)
    auprc = average_precision_score(y_true=labels, y_score=preds)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=labels, y_score=preds)

    # Specificity = 1 - false_positive_rate
    # Selecting false_positive_rate below 0.05!
    under_005_indices = np.where(false_positive_rate <= 0.05)[0]
    sensitivity_at_95_perc_spec = false_positive_rate[under_005_indices[-1]] if len(under_005_indices) > 0 else 0.0
    return {
        'AUROC': auroc,
        'AUPRC': auprc,
        'sens_at_95_spec': sensitivity_at_95_perc_spec
    }