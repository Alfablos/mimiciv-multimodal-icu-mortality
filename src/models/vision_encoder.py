import torch
from torch import nn, Tensor
import torchvision.models as visionmodels
from torchvision.models import DenseNet


class Xencoder(nn.Module):
    def __init__(self, dropout=0.3):
        super(Xencoder, self).__init__()
        
        self.backbone: DenseNet = visionmodels.densenet121(weights=visionmodels.DenseNet121_Weights.DEFAULT)

        for name, param in self.backbone.named_parameters():
            if "denseblock4" not in name and 'norm5' not in name and 'classifier' not in name:
                param.requires_grad = False
        self.backbone.classifier = nn.Identity()


        self.head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=256) # encoding vector
        )
        
    def forward(self, x) -> Tensor:
        x: Tensor = self.backbone(x)
        yh: Tensor = self.head(x)
        return yh


if __name__ == '__main__':
    enc = Xencoder().to('cuda')
    img_batch = torch.randn(size=(5, 3, 512, 512), device='cuda', dtype=torch.float32)
    output = enc(img_batch)
    assert output.shape == (5, 256) # (batch_size, 256)