import torch
from tqdm import tqdm
from dataset import get_dataloaders
from model import SimCLRModel
from loss import nt_xent_loss
from config import config

def train():
    # Force CPU
    device = torch.device('cpu')
    print(f"Training on {device}")

    model = SimCLRModel(config["model_name"], config["projection_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    dataloader = get_dataloaders(config)

    for epoch in range(config["epochs"]):
        total_loss = 0.0
        model.train()
        for (x1, x2) in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1), model(x2)

            loss = nt_xent_loss(z1, z2, config["temperature"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['epochs']}, Avg Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"simclr_epoch{epoch+1}.pth")
