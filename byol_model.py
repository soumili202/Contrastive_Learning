import torch
import torch.nn as nn
import torchvision.models as models

class BYOLModel(nn.Module):
    def __init__(self, base_encoder='resnet18', projection_dim=64, momentum=0.996):
        super(BYOLModel, self).__init__()
        self.online_encoder = models.__dict__[base_encoder](pretrained=False)
        self.online_encoder.fc = nn.Identity()

        self.target_encoder = models.__dict__[base_encoder](pretrained=False)
        self.target_encoder.fc = nn.Identity()

        self.online_projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
        self.target_projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

        self._momentum_update_target_encoder()
        self.momentum = momentum

    def _momentum_update_target_encoder(self):
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False
        for param_online, param_target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False

    def forward(self, x1, x2):
        o1 = self.online_predictor(self.online_projector(self.online_encoder(x1)))
        o2 = self.online_predictor(self.online_projector(self.online_encoder(x2)))

        with torch.no_grad():
            self._momentum_update_target_encoder()
            t1 = self.target_projector(self.target_encoder(x1))
            t2 = self.target_projector(self.target_encoder(x2))

        return o1, t2.detach(), o2, t1.detach()
