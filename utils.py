import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, images, labels, transforms, num_classes=2):
        self.images = images
        self.labels = labels
        self.num_classes = num_classes
        self.transforms = transforms

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.images)

    def __getitem__(self, index):
        x = self.images[index][:,:,:3]
        y = self.labels[index]
        return self.transforms(x), y


def custom_collate_fn(data):
    x = list(map(lambda x: x[0], data))
    y = list(map(lambda x: torch.Tensor(x[1]), data))
    x = torch.stack(x)
    y = torch.stack(y).squeeze().long()
    return x, y


def graph_losses(losses):
    for phase, color in zip(['train', 'eval'], ['r--', 'b--']):
        if not losses[phase]:
            continue
        epoch_count = range(1, len(losses[phase]) + 1)
        print(losses[phase])
        plt.plot(epoch_count, losses[phase], color)
        plt.legend([f'{phase.capitalize()} Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
