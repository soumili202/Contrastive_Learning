import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from byol_model import BYOLModel
from config import config

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_eval_dataloader():
    transform = T.Compose([
        T.Resize((config["image_size"], config["image_size"])),
        T.ToTensor()
    ])
    dataset = STL10(root="./data", split="train", transform=transform, download=True)
    subset = Subset(dataset, range(1000))  # faster evaluation
    return DataLoader(subset, batch_size=64, shuffle=False, num_workers=0)

def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            z = model.online_encoder(x)  # Only online encoder (don't use predictor here)
            embeddings.append(z.cpu())
            labels.append(y)
    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()

def plot_tsne(embeddings, labels, save_path="tsne_byol.png"):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap='tab10', s=10)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE of BYOL Learned Embeddings")
    plt.savefig(save_path)
    plt.show()

def main():
    device = get_device()
    print(f"Evaluating BYOL on {device}")

    model = BYOLModel(base_encoder=config["model_name"], projection_dim=config["projection_dim"])
    model.load_state_dict(torch.load("byol_model/byol_epoch10.pth", map_location=device))
    model.to(device)

    dataloader = get_eval_dataloader()
    embeddings, labels = extract_embeddings(model, dataloader, device)
    plot_tsne(embeddings, labels)

if __name__ == "__main__":
    main()
