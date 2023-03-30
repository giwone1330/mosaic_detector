import cv2
import os
import pathlib
import PIL
import glob
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_paths = sorted(os.listdir(folder_path))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_paths[idx])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, image_path


class mosaic_detector(nn.Module):
    def __init__(self, mosaic_size):
        super(mosaic_detector, self).__init__()

        self.samecolor = nn.Conv2d(3, 9, mosaic_size, 1, 1, bias=False)
        self.negval = 1 - (mosaic_size*mosaic_size)

        # initialize conv to designed filter
        self.samecolor.weight.data.copy_(torch.ones(
            (9, 3, 3, 3), requires_grad=False, dtype=torch.float))
        for i in range(mosaic_size):
            for j in range(mosaic_size):
                self.samecolor.weight.data[mosaic_size *
                                           i+j, :, i, j] = self.negval

    def forward(self, x):
        x = self.samecolor(x)
        x_max, _ = torch.max(x, dim=1)
        x_min, _ = torch.min(x, dim=1)
        return x_max-x_min


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_dir = "./images"
    mosaic_size = 3

    model = mosaic_detector(mosaic_size=mosaic_size)
    model.to(device)
    batch_size = 1

    # Example usage
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # only one epoch
    for i, (img, path) in enumerate(dataloader):
        # load to gpu
        area = img.shape[-2]*img.shape[-1]
        img = img.to(device)

        # forward
        out = model(img).squeeze()  # out.shape = (w, h)
        zeros = torch.eq(out, 0.0)
        num_zeros = torch.sum(zeros).item()
        same_ratio = num_zeros / area
        print(same_ratio, path)


if __name__ == "__main__":
    main()
