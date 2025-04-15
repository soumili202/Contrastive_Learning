import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import STL10
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from model import SimCLRModel
from config import config

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_eval_dataloader():
    transform = T.Compose([
        T.Resize((config["image_size"], config["image_size"])),
        T.ToTensor()
    ])

    eval_set = STL10(root="./data", split="train", transform=transform, download=True)
    eval_subset = Subset(eval_set, range(1000))  # use 1k samples for speed
    return DataLoader(eval_subset, batch_size=64, shuffle=False, num_workers=0)

def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            z = model.encoder(x)  # only use encoder, skip projection head
            embeddings.append(z.cpu())
            labels.append(y)

    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()

def plot_tsne(embeddings, labels, save_path="tsne_plot.png", show_plot=False):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10)
    plt.legend(*scatter.legend_elements(), title="Classes", loc="best")
    plt.title("t-SNE of SimCLR Learned Embeddings")
    plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close()

def main():
    device = get_device()
    print(f"Evaluating on device: {device}")

    model = SimCLRModel(config["model_name"], config["projection_dim"])
    model.load_state_dict(torch.load("simclr_epoch69.pth", map_location=device))
    model.to(device)

    dataloader = get_eval_dataloader()
    embeddings, labels = extract_embeddings(model, dataloader, device)
    plot_tsne(embeddings, labels)

if __name__ == "__main__":
    main()
