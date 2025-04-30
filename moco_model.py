import torch
import torch.nn as nn
import torchvision.models as models

class MoCoModel(nn.Module):
    def __init__(self, base_encoder='resnet18', projection_dim=64, queue_size=65536, momentum=0.999):
        super(MoCoModel, self).__init__()
        self.encoder_q = models.__dict__[base_encoder](pretrained=False)
        self.encoder_q.fc = nn.Identity()
        self.encoder_k = models.__dict__[base_encoder](pretrained=False)
        self.encoder_k.fc = nn.Identity()

        self.projector_q = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
        self.projector_k = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.momentum = momentum

        self._momentum_update_key_encoder()

    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr+batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue.size(1)
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = nn.functional.normalize(self.projector_q(self.encoder_q(im_q)), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = nn.functional.normalize(self.projector_k(self.encoder_k(im_k)), dim=1)
        return q, k
