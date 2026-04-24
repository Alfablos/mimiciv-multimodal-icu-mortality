import torch.nn.functional as F
import torchvision.transforms.functional as VF
from enum import Enum
import torch
from torch import Tensor, nn
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.fusion import Fusion
from data import MIMICReduced
from config import (
    image_extension,
    image_base_dir,
    hyperparameters
)

activations: Tensor | None
gradients: Tensor | None

class Architecture(Enum):
    DENSENET121 = 'densenet121'
    RESNET18 = 'resnet18'
    
    def gradcam_layer(self, model: nn.Module):
        if self == Architecture.DENSENET121:
            return model.vision_encoder.backbone.features.denseblock4
        else:
            raise NotImplementedError('Grad-CAM layer extraction not yet implemented for Resnet18')
    

def collect_activations(module, input, output): # forward hook
    global activations
    activations = output.detach()

def collect_gradients(module, gradient_input, gradient_output: tuple[Tensor, ...]): # backward hook
    global gradients
    gradients = gradient_output[0].detach()

def trace_activations() -> tuple[Tensor, Tensor, Tensor]:
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    model: Fusion = mlflow.pytorch.load_model('runs:/51fe9faf09c5470eb5a5a420bd1d03e3/multimodal_icu_mortality')
    target_layer = model.vision_encoder_gradcam_layer()
    fhook = target_layer.register_forward_hook(collect_activations)
    bhook = target_layer.register_full_backward_hook(collect_gradients)
    ds = MIMICReduced(
        df=pd.read_csv('./ds_val.csv'),
        label_column='hospital_expire_flag',
        images_extension=image_extension,
        images_base_dir=image_base_dir,
        debug=True,
        # optional for the validation set as well but
        # allowes me to iterate faster
        limit=hyperparameters['train_limit']
    )
    original_image, tab, _ = ds[10]
    tab, image = tab.to(device), original_image.to(device)
    tab = tab.unsqueeze(dim=0) # pytorch interprets as dim 0 the batch size. We go from [34] to [1, 34]
    image = MIMICReduced.gpu_transforms(image.unsqueeze(dim=0)) # batch of 1
    model.to(device)
    preds = model(image, tab) # current shape: (1, 1)
    
    preds[0, 0].backward() # to trigger backprop => generating gradients the backward hook can capture
    
    fhook.remove()
    bhook.remove()

    return activations, gradients, original_image




if __name__ == '__main__':
    # Activations: A1...A1024
    # Gradients: dy/dA
    activations, gradients, original_image = trace_activations()
    original_image = VF.to_pil_image(original_image)
    # Shape for both should now be (batch_size, C, H, W)
    # batch_size = 1 because we're only passing 1 image and unsqueezing the tensor
    # other dimensions are those of denseblock4 (Densenet: C=1024, H=16, W=16
    print('Activations shape: ', activations.shape)
    print('Gradients shape: ', gradients.shape)
    
    assert activations.shape == gradients.shape, "Activations and gradients should have the same shape!"
    
    # Average influence of each part of HxW (in A) on y
    # The mean is computer PER CHANNEL, so batch_size an C (num channels) are left intact: (1, 1024, 1, 1) is the new shape on Densenet121.
    # This tells how much each of the 1024 16x16 layers impacts y (binary, single class, death or not)
    # Equivalent to ( 1/(HxW) ) * sum_for_i(sum_for_j( dy/dAij )) for a single class
    importance_weights = gradients.mean(dim=(2, 3), keepdim=True)
    print('Importance weights shape:', importance_weights.shape)
    
    # Linear combination of importance weights * activations
    # Now A is represented based on how much it influenced y,
    # then the 1024 channels will now be merged so we'll only have 16x16.
    # 1. linear combination
    heat = importance_weights * activations # => 1, C, H, W
    # 2. sum by channel
    heat = torch.sum(heat, dim=1, keepdim=True) # => 1, 1, H, W
    # But we only care for values > 0
    heat = F.relu(heat)
    
    # If we want to overlay the images they have to have the same dims
    heat = F.interpolate(heat, size=(512, 512), mode='bilinear', align_corners=False)
    heat = heat.squeeze().cpu().numpy()
    # and normalize the heatmap too
    heat = ((heat - np.min(heat)) / (np.max(heat) + 1e-8))


    plt.figure(figsize=(10, 10))
    plt.imshow(original_image, cmap='gray')
    plt.imshow(heat, cmap='jet', alpha=0.3)
    plt.axis('off')
    plt.savefig('./gradcam.png', bbox_inches='tight', pad_inches=0, dpi=300)
    
    