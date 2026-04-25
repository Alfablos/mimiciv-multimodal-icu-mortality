import torch.nn.functional as F
from enum import Enum
import torch
from matplotlib.figure import Figure
from torch import Tensor, nn
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .models.fusion import Fusion
from .data import MIMICReduced
from .config import image_extension, image_base_dir, hyperparameters


class Architecture(Enum):
    DENSENET121 = "densenet121"
    RESNET18 = "resnet18"

    def gradcam_layer(self, model: nn.Module):
        if self == Architecture.DENSENET121:
            return model.vision_encoder.backbone.features.denseblock4
        else:
            raise NotImplementedError(
                "Grad-CAM layer extraction not yet implemented for Resnet18"
            )


def trace_activations(
    model: Fusion,
    image_tensor: Tensor,  # expects batch
    tab_tensor: Tensor,
    transform_images=True,
) -> tuple[Tensor, Tensor, Tensor]:
    activations: Tensor | None = None
    gradients: Tensor | None = None

    def collect_activations(module, input, output):  # forward hook
        nonlocal activations
        activations = output.detach()

    def collect_gradients(
        module, gradient_input, gradient_output: tuple[Tensor, ...]
    ):  # backward hook
        nonlocal gradients
        gradients = gradient_output[0].detach()

    device = next(model.parameters()).device
    target_layer = model.gradcam_layer()
    fhook = target_layer.register_forward_hook(collect_activations)
    bhook = target_layer.register_full_backward_hook(collect_gradients)

    tab_tensor, image_tensor = tab_tensor.to(device), image_tensor.to(device)
    if transform_images:
        image_tensor = MIMICReduced.gpu_transforms(image_tensor)  # batch of 1

    with torch.enable_grad():
        preds = model(image_tensor, tab_tensor)  # current shape: (1, 1)
        model.zero_grad()
        # Autograd works ON SINGLE SCALARS!!
        preds[
            0, 0
        ].backward()  # to trigger backprop => generating gradients the backward hook can capture

    fhook.remove()
    bhook.remove()

    return activations, gradients, image_tensor


def grad_cam(
    model: nn.Module,
    image_tensor: Tensor,  # Expects a tensor of 1, no more!
    tab_tensor: Tensor,
    transform_images=True,
) -> Figure:
    if not len(image_tensor) == len(tab_tensor):
        raise ValueError(
            "ERROR: GradCAM function needs images, tabs to have the same size"
        )
    if not len(image_tensor) == 1:
        raise ValueError(
            "the grad_cam function only accepts 1 as the first dimension of the images tensor. This is a bug."
        )
    activations, gradients, original_image = trace_activations(
        model, image_tensor, tab_tensor, transform_images
    )
    # Shape for both should now be (batch_size, C, H, W)
    # batch_size = 1 because we're only passing 1 image and unsqueezing the tensor
    # other dimensions are those of denseblock4 (Densenet: C=1024, H=16, W=16
    # print('Activations shape: ', activations.shape)
    # print('Gradients shape: ', gradients.shape)

    assert activations.shape == gradients.shape, (
        "Activations and gradients should have the same shape!"
    )

    # Average influence of each part of HxW (in A) on y
    # The mean is computer PER CHANNEL, so batch_size an C (num channels) are left intact: (1, 1024, 1, 1) is the new shape on Densenet121.
    # This tells how much each of the 1024 16x16 layers impacts y (binary, single class, death or not)
    # Equivalent to ( 1/(HxW) ) * sum_for_i(sum_for_j( dy/dAij )) for a single class
    importance_weights = gradients.mean(dim=(2, 3), keepdim=True)
    # print('Importance weights shape:', importance_weights.shape)

    # Linear combination of importance weights * activations
    # Now A is represented based on how much it influenced y,
    # then the 1024 channels will now be merged so we'll only have 16x16.
    # 1. linear combination
    heat = importance_weights * activations  # => 1, C, H, W
    # 2. sum by channel
    heat = torch.sum(heat, dim=1, keepdim=True)  # => 1, 1, H, W
    # But we only care for values > 0
    heat = F.relu(heat)

    # If we want to overlay the images they have to have the same dims
    heat = F.interpolate(
        heat, size=(512, 512), mode="bilinear", align_corners=False
    )  # no transpose afterwards!
    heat = heat.squeeze().cpu().numpy()
    # and normalize the heatmap too
    heat = (heat - np.min(heat)) / (np.max(heat) + 1e-8)

    original_image = original_image.squeeze().cpu().numpy()
    original_image = (original_image - np.min(original_image)) / (
        np.max(original_image) + 1e-8
    )
    original_image = np.transpose(original_image, (1, 2, 0))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(original_image, cmap="gray")
    ax.imshow(heat, cmap="jet", alpha=0.5)
    ax.axis("off")

    return fig


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    model: Fusion = mlflow.pytorch.load_model(
        "runs:/51fe9faf09c5470eb5a5a420bd1d03e3/multimodal_icu_mortality"
    )

    ds = MIMICReduced(
        df=pd.read_csv("./ds_val.csv"),
        label_column="hospital_expire_flag",
        images_extension=image_extension,
        images_base_dir=image_base_dir,
        debug=True,
        # optional for the validation set as well but
        # allowes me to iterate faster
        limit=hyperparameters["train_limit"],
    )

    # Activations: A1...A1024
    # Gradients: dy/dA
    image_t, tab_t, _ = ds[0:1]  # keeps dimes to [1, C, H, W] for the image
    fig = grad_cam(
        image_tensor=image_t, tab_tensor=tab_t, model=model, transform_images=True
    )

    plt.savefig("./gradcam.png", bbox_inches="tight", pad_inches=0, dpi=300)
