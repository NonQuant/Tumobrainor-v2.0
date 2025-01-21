import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TumobrainorDataset(Dataset):
    def __init__(self, images, labels) -> None:
        self.X = images
        self.y = labels
        # function for images transformations
        self.random_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=45),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        num_augment = 8
        augmented_batch = []
        # creating augmented data
        for i in range(num_augment):
            new_item = self.random_transform(self.X[index])
            augmented_batch.append(new_item)
        # labels with one-hot encoding
        labels = torch.zeros(4, dtype=torch.float32)
        labels[int(self.y[index]) - 1] = 1

        new_labels = [labels, labels, labels, labels, labels, labels, labels, labels]

        return torch.stack(augmented_batch), torch.stack(new_labels)
