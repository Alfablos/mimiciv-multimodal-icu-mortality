import torch
from torch import Tensor
import mlflow
import pandas as pd

from models.fusion import Fusion
from data import MIMICReduced
from config import (
    image_extension,
    image_base_dir,
    hyperparameters
)

activations: Tensor | None
gradients: Tensor | None


def collect_activations(module, input, output): # forward hook
    global activations
    activations = output.detach()

def collect_gradients(module, gradient_input, gradient_output: tuple[Tensor, ...]): # backward hook
    global gradients
    gradients = gradient_output[0].detach()

def trace_activations() -> tuple[Tensor, Tensor]:
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    model: Fusion = mlflow.pytorch.load_model('runs:/51fe9faf09c5470eb5a5a420bd1d03e3/multimodal_icu_mortality')
    target_layer = model.vision_encoder.backbone.features.denseblock4
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
    image, tab, _ = ds[0]
    tab, image = tab.to(device), image.to(device)
    tab = tab.unsqueeze(dim=0) # pytorch interprets as dim 0 the batch size. We go from [34] to [1, 34]
    image = MIMICReduced.gpu_transforms(image.unsqueeze(dim=0)) # batch of 1
    model.to(device)
    preds = model(image, tab) # current shape: (1, 1)
    
    preds[0, 0].backward() # to trigger backprop => generating gradients the backward hook can capture
    
    fhook.remove()
    bhook.remove()

    return activations, gradients




if __name__ == '__main__':
    activations, gradients = trace_activations()
    print('Activations shape: ', activations.shape)
    print('Gradients shape: ', gradients.shape)
    