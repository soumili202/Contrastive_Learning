import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    similarity = torch.mm(z, z.T)
    n = z1.size(0)
    mask = torch.eye(2*n, device=z.device).bool()
    positives = torch.cat([torch.diag(similarity, n), torch.diag(similarity, -n)])
    negatives = similarity[~mask].view(2*n, -1)
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2*n, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits / temperature, labels)
