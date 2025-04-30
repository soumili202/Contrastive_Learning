import torch
from tqdm import tqdm
from dataset import get_dataloaders
from moco_model import MoCoModel
from moco_loss import moco_loss
from config import config
from pathlib import Path

MODEL_SAVE_PATH = Path("moco_model")


def train_moco():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training MoCo on {device}")

    model = MoCoModel(
        base_encoder=config["model_name"], projection_dim=config["projection_dim"]
    )
    # if model is already saved, load the last existing weight
    # if not, start from scratch
    latest_model = None
    first_epoch = 1
    for saved_model in MODEL_SAVE_PATH.glob("moco_epoch*.pth"):
        epoch = int(str(saved_model).split("moco_epoch")[1].split(".")[0]) + 1
        if epoch > first_epoch:
            latest_model = saved_model
            first_epoch = epoch
    if latest_model:
        print(f"Loading model from {latest_model}")
        model.load_state_dict(torch.load(latest_model, map_location=device))
        first_epoch = int(str(latest_model).split("moco_epoch")[1].split(".")[0]) + 1
    else:
        print("No saved model found, starting from scratch.")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    dataloader = get_dataloaders(config)

    for epoch in range(first_epoch - 1, config["epochs"]):
        total_loss = 0.0
        model.train()

        for x1, x2 in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            x1, x2 = x1.to(device), x2.to(device)
            q, k = model(x1, x2)

            loss = moco_loss(q, k, model.queue, temperature=config["temperature"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update queue
            with torch.no_grad():
                model.dequeue_and_enqueue(k)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['epochs']} | Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), MODEL_SAVE_PATH / f"moco_epoch{epoch+1}.pth")


if __name__ == "__main__":
    train_moco()
