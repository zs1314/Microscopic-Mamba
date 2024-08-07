import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = self.get_image_paths()
        self.transform = transform

    def get_image_paths(self):
        image_paths = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(subdir, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def calculate_mean_std(dataloader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_images = 0

    for images in dataloader:
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_images += batch_size

    mean /= n_images
    std /= n_images

    return mean, std


def main():
    # replace your dataset path
    root_dir = r"your dataset path"
    """    
        --data
          -train
            ...
          -val
            ...
          -test
            ...
    """
    batch_size = 72
    num_workers = 10

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = ImageDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean, std = calculate_mean_std(dataloader)

    print('Mean:', mean)
    print('Std:', std)


if __name__ == '__main__':
    main()
