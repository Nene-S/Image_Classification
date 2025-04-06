import torch
import pathlib
import os
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ImageDataset(Dataset):
    """
    Custom dataset for loading images and corresponding labels.

    Attributes:
        file_list (list): List of image file paths.
        label (list): List of corresponding labels.
        transform (callable, optional): A function/transform to apply to the images.
    """

    def __init__(self, file_list, label, transform=None):
        """
        Initializes the ImageDataset.

        Args:
            file_list (list of str): List of file paths to images.
            label (list of int): List of corresponding labels for the images.
            transform (callable, optional): A function/transform to apply to the images. Defaults to None.
        """
        self.file_list = file_list
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        """
        Loads an image from file and applies transformations.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: Transformed image tensor and label tensor.
        """
        img = Image.open(self.file_list[index])
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(self.label[index], dtype=torch.int64).to(device)
        return img.to(device), label

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.label)

def compute_mean_std(dataset, channels=3):
    """
    Computes the mean and standard deviation of a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to compute statistics for.

    Returns:
        tuple: (mean, std) for each channel.
    """
    loader = DataLoader(dataset, batch_size=60, shuffle=False)
    mean = torch.zeros(channels, device=device) 
    std = torch.zeros(channels, device=device)
    total_images = 0

    for images, _ in loader:
        batch_size, _, _, _ = images.shape
        total_images += batch_size
        mean += images.mean(dim=[0, 2, 3]) * batch_size
        std += images.std(dim=[0, 2, 3]) * batch_size

    mean /= total_images
    std /= total_images
    mean = torch.round(mean * 10000) / 10000
    std = torch.round(std * 10000) / 10000

    return mean.tolist(), std.tolist()

def load_images_from_folder(path):
    """
    Loads image file paths and assigns numeric labels based on folder names.

    Args:
        path (str): Path to the dataset directory.

    Returns:
        tuple: (file_list, label_list) containing file paths and corresponding labels.
    """
    file_list = []
    label_list = []

    for idx, folder in enumerate(sorted(os.listdir(path))):
        cur_fld = pathlib.Path(os.path.join(path, folder))
        for img_path in cur_fld.glob("*.jpg"):
            file_list.append(str(img_path))
            label_list.append(idx)

    return file_list, label_list

def load_dataset():
    """
    Loads the training and testing datasets, computes normalization statistics, and 
    applies the necessary transformations.

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    
    train_file_list, train_label = load_images_from_folder(config.train_path)
    test_file_list, test_label = load_images_from_folder(config.test_path)

    train_idx, valid_idx = train_test_split(np.arange(len(train_file_list)),
                                            test_size=0.2,
                                            stratify=train_label,
                                            random_state=22)

    img_height, img_width = 64, 64
    temp_transform = transforms.Compose([transforms.Resize((img_height, img_width)), transforms.ToTensor()])
    temp_dataset = ImageDataset(train_file_list, train_label, temp_transform)
    mean, std = compute_mean_std(temp_dataset)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
])
    test_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    full_train_dataset = ImageDataset(train_file_list, train_label, train_transform)
    test_dataset = ImageDataset(test_file_list, test_label, test_transform)

   
    train_dataset = Subset(full_train_dataset, train_idx)
    valid_dataset = Subset(ImageDataset(train_file_list, train_label, test_transform), valid_idx)

    return train_dataset, valid_dataset, test_dataset

def main():
    """
    Main function to visualize a sample image from the training dataset.
    """
    train, valid, test = load_dataset()
    print(len(train), len(valid), len(test))
    img, label = train[20]
    print(img.shape)
    plt.imshow(img.cpu().permute(1,2,0))  # Move image back to CPU for plotting
    labels_dict = {0: "happy", 1: "neutral", 2: "sad"}
    plt.title(f"Image Label: {labels_dict.get(label.item())}")
    plt.show()

if __name__ == "__main__":
    main()
