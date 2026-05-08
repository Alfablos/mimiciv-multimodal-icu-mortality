import torch
from torch import Tensor, nn

import mmim.trainer.models.fusion as fusion_module
from tests.unit.data import batch_size, current_features, images_shape


class FakeVisionEncoder(nn.Module):
    def __init__(self, encoding_vector_dims, **kwargs):
        super().__init__()
        self.encoding_vector_dims = encoding_vector_dims

    def forward(self, image):
        return torch.ones(image.shape[0], self.encoding_vector_dims)


def test_model_forward_outputs_right_shapes(monkeypatch):
    monkeypatch.setattr(fusion_module, "Xencoder", FakeVisionEncoder)
    model = fusion_module.Fusion()

    images = torch.randn(
        size=(batch_size, *images_shape), dtype=torch.float32
    ).transpose(1, 3)
    tabs = torch.randn(batch_size, len(current_features))
    logits: Tensor = model(images, tabs)

    assert logits.shape == (batch_size, 1)
