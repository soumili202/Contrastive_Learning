import torch.nn as nn
import torchvision.models as models

class SimCLRModel(nn.Module):
    def __init__(self, base_model, projection_dim):
        super().__init__()
        self.encoder = models.__dict__[base_model](pretrained=False)
        self.encoder.fc = nn.Identity()  # Remove classifier
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z
