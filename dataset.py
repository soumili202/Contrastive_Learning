import torchvision.transforms as T
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, Subset
from PIL import Image

class SimCLRDataset(STL10):
    def __init__(self, split, transform=None):
        super().__init__(root='./data', split=split, download=True)
        self.transform = transform

    def __getitem__(self, index):
        img, _ = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        return self.transform(img), self.transform(img)

    def __len__(self):
        return len(self.data)

def get_dataloaders(config):
    augment = T.Compose([
        T.RandomResizedCrop(config["image_size"]),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=5),
        T.ToTensor()
    ])

    # STL-10 split = 'unlabeled' for contrastive training
    dataset = SimCLRDataset(split='unlabeled', transform=augment)
    dataset = Subset(dataset, range(5000))  # Use only first 5k images for CPU training

    dataloader = DataLoader(dataset, batch_size=config["batch_size"],
                            shuffle=True, num_workers=0)

    return dataloader
