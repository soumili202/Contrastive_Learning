import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
import torchvision.transforms as T
from model import SimCLRModel
from config import config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from pathlib import Path
MODEL_SAVE_PATH = Path("simclr_model")

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
    embeddings = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            z = model.encoder(x)
            embeddings.append(z.cpu())
            labels.append(y)
    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()

def main():
    device = get_device()
    print(f"Running linear evaluation on: {device}")

    model = SimCLRModel(config["model_name"], config["projection_dim"])
    model.load_state_dict(torch.load(MODEL_SAVE_PATH / "simclr_epoch100.pth", map_location=device))
    model.to(device)

    # Get training data (supervised labels)
    train_loader = get_dataloader(split="train")
    X_train, y_train = extract_embeddings(model, train_loader, device)

    # Get test data
    test_loader = get_dataloader(split="test")
    X_test, y_test = extract_embeddings(model, test_loader, device)

    # Train linear classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Linear evaluation accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()

