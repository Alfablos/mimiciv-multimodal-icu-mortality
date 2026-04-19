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

        for images, tabs, labels in train_loader:
            images, tabs, labels = images.to(device), tabs.to(device), labels.unsqueeze(dim=1).to(device)
            images = MIMICReduced.gpu_transforms(images)

            optimizer.zero_grad()
            preds = model(images, tabs)
            loss = loss_fn(preds, labels)
            loss.backward()
            print(f'Epoch {epoch} train loss:', loss.item())
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for val_images, val_tabs, val_labels in val_loader:
                val_images, val_tabs, val_labels = val_images.to(device), val_tabs.to(device), val_labels.unsqueeze(dim=1).to(device)
                val_images = MIMICReduced.gpu_transforms(val_images)

                val_preds = model(val_images, val_tabs)
                val_loss = loss_fn(val_preds, val_labels).item()
                print(f'Validation loss for epoch {epoch}:', val_loss)
