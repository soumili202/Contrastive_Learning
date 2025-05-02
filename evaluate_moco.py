import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from moco_model import MoCoModel
from config import config

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_eval_dataloader():
    transform = T.Compose([
        T.Resize((config["image_size"], config["image_size"])),
        T.ToTensor()
    ])
    dataset = STL10(root="./data", split="train", transform=transform, download=True)
    subset = dataset
    return DataLoader(subset, batch_size=64, shuffle=False, num_workers=0)

def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            z = model.encoder_q(x)  # Only query encoder
            embeddings.append(z.cpu())
            labels.append(y)
    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()

"""
    plt.figure(figsize=(10*2,8*2))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=25)
    plt.legend(*scatter.legend_elements(), title="Classes", loc="best", fontsize=25, title_fontsize=30)
    plt.colorbar(scatter, label="Classes", fontsize=25)
    plt.title("t-SNE of SimCLR Learned Embeddings", fontdict={"fontsize": 40})
    plt.xlabel("t-SNE Component 1", fontsize=30)
    plt.ylabel("t-SNE Component 2", fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    
"""

def plot_tsne(embeddings, labels, save_path="tsne_moco.png"):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10*2,8*2))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap='tab10', s=25)
    plt.legend(*scatter.legend_elements(), title="Classes", loc="best", fontsize=25, title_fontsize=30)
    plt.title("t-SNE of MoCo Learned Embeddings", fontdict={"fontsize": 40})
    plt.xlabel("t-SNE Component 1", fontsize=30)
    plt.ylabel("t-SNE Component 2", fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()
    plt.savefig(save_path)

def main():
    device = get_device()
    print(f"Evaluating MoCo on {device}")

    model = MoCoModel(base_encoder=config["model_name"], projection_dim=config["projection_dim"])
    model.load_state_dict(torch.load("moco_model/moco_epoch50.pth", map_location=device))
    model.to(device)

    dataloader = get_eval_dataloader()
    embeddings, labels = extract_embeddings(model, dataloader, device)
    plot_tsne(embeddings, labels)

if __name__ == "__main__":
    main()
