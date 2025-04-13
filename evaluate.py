from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

def plot_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE of Learned Embeddings")
    plt.savefig("tsne_embeddings.png")
