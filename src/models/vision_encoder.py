import torch
from torch import nn, Tensor
import torchvision.models as visionmodels
from torchvision.models import DenseNet

DEFAULT_WEIGHTS = visionmodels.DenseNet121_Weights.DEFAULT

class Xencoder(nn.Module):
    def __init__(self, encoding_vector_dims, frozen_backbone, dropout):
        super().__init__()
        
        self.backbone: DenseNet = visionmodels.densenet121(weights=visionmodels.DenseNet121_Weights.DEFAULT)

        if frozen_backbone is None:
            raise ValueError('A value for `frozen_backbone` MUST be specified for the X-Ray encoder.')

        if frozen_backbone:
            for name, param in self.backbone.named_parameters():
                if "denseblock4" not in name and 'norm5' not in name and 'classifier' not in name:
                    param.requires_grad = False

        self.backbone.classifier = nn.Identity()


        self.head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=encoding_vector_dims) # encoding vector
        )
        
    def forward(self, x) -> Tensor:
        x: Tensor = self.backbone(x)
        return  self.head(x)



if __name__ == '__main__':
    enc = Xencoder().to('cuda')
    img_batch = torch.randn(size=(5, 3, 512, 512), device='cuda', dtype=torch.float32)
    output = enc(img_batch)
    assert output.shape == (5, 256) # (batch_size, 256)