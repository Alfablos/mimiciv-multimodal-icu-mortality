from functorch.dim import Tensor
import torch
from trainer.models.fusion import Fusion

from data import current_features, batch_size, images_shape


def test_model_forward_outputs_right_shapes():
    model = Fusion()

    images = torch.randn(
        size=(batch_size, *images_shape), dtype=torch.float32
    ).transpose(1, 3)
    tabs = torch.randn(batch_size, len(current_features))
    logits: Tensor = model(images, tabs)

    assert logits.shape == (batch_size, 1)
