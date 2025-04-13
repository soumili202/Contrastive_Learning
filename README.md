# SimCLR - Contrastive Learning Framework

This repository implements a simplified version of the SimCLR framework for contrastive learning using the STL-10 dataset. The project is designed to train a neural network to learn meaningful representations of images without requiring labeled data.

## Features

- **SimCLR Framework**: Implements the SimCLR contrastive learning approach.
- **STL-10 Dataset**: Uses the STL-10 dataset for training and evaluation.
- **Custom Data Augmentation**: Includes a pipeline for strong data augmentations.
- **t-SNE Visualization**: Visualizes learned embeddings using t-SNE.
- **Configurable Training**: Easily configurable training parameters via `config.py`.

## Project Structure

```
.
├── config.py               # Configuration for training parameters
├── dataset.py              # Dataset and data augmentation pipeline
├── evaluate.py             # t-SNE visualization of learned embeddings
├── loss.py                 # NT-Xent loss implementation
├── model.py                # SimCLR model definition
├── train.py                # Training script
├── data/                   # Contains STL-10 dataset and related files
├── simclr_epoch*.pth       # Saved model checkpoints
└── README.md               # Project documentation
```

## Requirements

- Python 3.10 or later
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- tqdm

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/soumili202/Contrastive_Learning.git
   cd Contrastive_Learning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the STL-10 dataset:
   The dataset will be automatically downloaded when running the training script.

## Usage

### Training

To train the SimCLR model, run the train.py script:

```bash
python train.py
```

This will train the model for the number of epochs specified in config.py and save checkpoints (`simclr_epoch*.pth`) after each epoch.

### Visualization

To visualize the learned embeddings using t-SNE, run the evaluate.py script:

```bash
python evaluate.py
```

This will generate a `tsne_embeddings.png` file showing the t-SNE plot of the embeddings.

### Configuration

Modify the config.py file to adjust training parameters such as batch size, learning rate, number of epochs, and more.

Example configuration:
```python
config = {
    "batch_size": 32,
    "image_size": 64,
    "epochs": 10,
    "temperature": 0.5,
    "lr": 3e-4,
    "dataset": "STL10",
    "model_name": "resnet18",
    "embedding_dim": 128,
    "projection_dim": 64
}
```

## File Descriptions

- **`config.py`**: Contains training parameters such as batch size, learning rate, and model configuration.
- **`dataset.py`**: Defines the `SimCLRDataset` class for loading and augmenting the STL-10 dataset.
- **`train.py`**: Implements the training loop for the SimCLR model.
- **`loss.py`**: Implements the NT-Xent loss function for contrastive learning.
- **`model.py`**: Defines the SimCLR model architecture, including the encoder and projection head.
- **`evaluate.py`**: Visualizes the learned embeddings using t-SNE.

## Results

- The model checkpoints (`simclr_epoch*.pth`) are saved after each epoch.
- The t-SNE visualization (`tsne_embeddings.png`) provides insights into the quality of the learned embeddings.
![t-SNE Visualization](tsne_embeddings.png)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- PyTorch and torchvision libraries
- STL-10 dataset
