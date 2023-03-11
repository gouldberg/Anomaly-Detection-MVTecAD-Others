
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import pandas as pd
import torchvision
from torchvision.io import read_image

from torch.utils.data import DataLoader
import numpy as np


#########################################################################################################################
# -----------------------------------------------------------------------------------------------------------------------
# FasionMNIST dataset
# -----------------------------------------------------------------------------------------------------------------------

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# -----------------------------------------------------------------------------------------------------------------------
# sample images
# -----------------------------------------------------------------------------------------------------------------------

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# -----------------------------------------------------------------------------------------------------------------------
# creating custom dataset
# -----------------------------------------------------------------------------------------------------------------------

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# -----------------------------------------------------------------------------------------------------------------------
# data loader
# -----------------------------------------------------------------------------------------------------------------------

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


smpl = next(iter(train_dataloader))

# (64, 1, 28, 28)
print(smpl[0].shape)
print(smpl[1].shape)


# -----------------------------------------------------------------------------------------------------------------------
# iterating through data loader
# -----------------------------------------------------------------------------------------------------------------------

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")


#########################################################################################################################
# -----------------------------------------------------------------------------------------------------------------------
# CIFAR10
# -----------------------------------------------------------------------------------------------------------------------

class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, path, transform1=None, transform2=None, train=True):
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train

        self.labelset = torchvision.datasets.CIFAR10(root=path, train=self.train, download=True)
        self.dataset = torchvision.datasets.CIFAR10(root=path, train=self.train, download=True)

        self.datanum = len(dataset)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_label = self.labelset[idx][0]
        out_data = self.dataset[idx][0]

        if self.transform1:
            out_label = self.transform1(out_label)

        if self.transform2:
            out_data = self.transform2(out_data)

        return out_data, out_label


path_cifar10 = os.path.join(base_path, 'data', 'cifar10')
dataset = torchvision.datasets.CIFAR10(root=path_cifar10, train=True, download=True)

trans1 = torchvision.transforms.ToTensor()
trans2 = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])

idx = 0

image = dataset[idx][0]

# THIS IS PIL.Image.Image
print(image)

image_trans = trans2(image)
# torch.Size([1, 32, 32])
print(image_trans.shape)


# ----------
dataset = Mydatasets(path=path_cifar10, transform1=trans1, transform2=trans2, train=True)
trainloader = torch.utils.data.DataLoader(dataset, batch_size = 100, shuffle = True, num_workers = 2)
