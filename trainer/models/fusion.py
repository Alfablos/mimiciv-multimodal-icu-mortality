import torch
from torch import nn

from .vision_encoder import Xencoder
from .tabular_encoder import TabularEncoder


class Fusion(nn.Module):
    def __init__(
            self,
            tab_features_in=34,
            encoding_vector_dims=256,
            dropout=0.3,
            freeze_vision=True
    ):
        super().__init__()

        self.vision_encoder = Xencoder(
            frozen_backbone=freeze_vision,
            dropout=dropout,
            encoding_vector_dims=encoding_vector_dims
        )
        self.tabular_encoder = TabularEncoder(
            in_features=tab_features_in,
            encoding_vector_dims=encoding_vector_dims,
            dropout=dropout
        )

        fusion_vector_dims = 2 * encoding_vector_dims

        self.class_head = nn.Sequential(
            nn.Linear(in_features=fusion_vector_dims, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=1) # in-hospital death? (LOGITS!)
        )

    def forward(self, image, tab):
        v_vec = self.vision_encoder(image)
        t_vec = self.tabular_encoder(tab)

        f_vec = torch.cat([v_vec, t_vec], dim=1) # !!!
        return self.class_head(f_vec)
    
    def gradcam_layer(self):
        return self.vision_encoder.gradcam_layer()


if __name__ == '__main__':
    batch_size = 8

    fake_img = torch.randn(batch_size, 3, 512, 512)
    fake_vitals = torch.randn(batch_size, 34)
    model = Fusion()

    output = model(fake_img, fake_vitals)

    print('Vision embedding shape:', model.vision_encoder(fake_img).shape)
    print('Vitals embedding shape:', model.tabular_encoder(fake_vitals).shape)
    assert output.shape == (batch_size, 1), f'Expected output shape of ({batch_size}, 1), found {output.shape}.'