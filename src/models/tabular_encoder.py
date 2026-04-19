
from torch import nn





class TabularEncoder(nn.Module):
    def __init__(self, in_features, encoding_vector_dims, dropout):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, encoding_vector_dims),
            nn.ReLU() # Final activation before fusion
        )
    
    def forward(self, x):
        return self.layers(x)

