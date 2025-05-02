import torch
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import torchvision.transforms as T
from moco_model import MoCoModel
from config import config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from pathlib import Path

MODEL_SAVE_PATH = Path("moco_model")

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_dataloader(split):
    transform = T.Compose([
        T.Resize((config["image_size"], config["image_size"])),
        T.ToTensor()
    ])
    dataset = STL10(root="./data", split=split, transform=transform, download=True)
    return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            z = model.encoder_q(x)
            embeddings.append(z.cpu())
            labels.append(y)
    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()

def main():
    device = get_device()
    print(f"Running MoCo linear evaluation on: {device}")

    model = MoCoModel(config["model_name"], config["projection_dim"])
    model.load_state_dict(torch.load(MODEL_SAVE_PATH / "moco_epoch50.pth", map_location=device))
    model.to(device)

    train_loader = get_dataloader("train")
    test_loader = get_dataloader("test")

    X_train, y_train = extract_embeddings(model, train_loader, device)
    X_test, y_test = extract_embeddings(model, test_loader, device)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"MoCo Linear evaluation accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
