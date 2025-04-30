import torch
import torch.nn.functional as F

def moco_loss(q, k, queue, temperature=0.07):
    batch_size = q.shape[0]
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (batch_size, 1)
    l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])  # (batch_size, queue_size)

    logits = torch.cat([l_pos, l_neg], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long).to(q.device)

    logits /= temperature
    loss = F.cross_entropy(logits, labels)
    return loss
