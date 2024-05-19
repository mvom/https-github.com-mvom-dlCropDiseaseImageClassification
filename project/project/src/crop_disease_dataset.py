# -*- coding: utf-8 -*-
"""crop_disease_dataset.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PfD66UFPJYeygruycABEi4yzNzrjjkfb
"""

import os
import random
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image, ImageFile

class CropDiseaseDataset(data.Dataset):
    def __init__(self, root_dir, train=True, validation=False, gray_scale=False, segmented=False):
        """Initializes a dataset containing images and labels."""
        super().__init__()
        self.gray_scale = gray_scale
        self.root_dir = root_dir
        self.train = train
        self.validation = validation
        self.segmented = segmented

        # Get all jpg images
        self.files = []
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            # Filter files based on the segmented condition
            if self.segmented:
                files = [os.path.join(class_dir, file) for file in os.listdir(class_dir) if file.endswith(".jpg")]
            else:
                files = [os.path.join(class_dir, file) for file in os.listdir(class_dir) if file.endswith(".JPG")]
            self.files.extend(files)

        random.shuffle(self.files)

        # Define the split ratios
        train_ratio = 0.65
        validation_ratio = 0.15

        self.classes = []
        for root, dirs, _ in os.walk(root_dir):
            for directory in dirs:
                self.classes.append(directory)

        # Split the dataset into train, validation, and test sets
        if self.train:
            num_train_files = int(len(self.files) * train_ratio)
            self.files = self.files[:num_train_files]
        elif self.validation:
            num_train_files = int(len(self.files) * train_ratio)
            num_validation_files = int(len(self.files) * validation_ratio)
            self.files = self.files[num_train_files:num_train_files + num_validation_files]
        else:
            num_train_files = int(len(self.files) * train_ratio)
            num_validation_files = int(len(self.files) * validation_ratio)
            self.files = self.files[num_train_files + num_validation_files:]

        # Remove all invalid files
        def is_valid(filename):
            """Check that a file is not corrupted, is in RGB, and has non-zero size"""
            if os.path.getsize(filename) == 0:
                return False
            valid_extensions = ['.jpg']
            _, ext = os.path.splitext(filename)
            if ext.lower() not in valid_extensions:
                return False
            try:
                with Image.open(filename, 'r') as img:
                    img.verify()
                    mode = img.mode
                    return mode == 'RGB'
            except:
                return False

        wrong = []
        for i in range(len(self.files)):
            if not is_valid(self.files[i]):
                wrong.append(i)
        self.files = np.delete(self.files, wrong)
        random.shuffle(self.files)

        if self.gray_scale:
            self.means, self.stds = self.calculate_mean_std_gray()
            self.means = np.array(self.means)
            self.stds = np.array(self.stds)
        else:
            self.means, self.stds = self.calculate_mean_std()

        self.transform = self.get_transform()

    def calculate_mean_std(self):
        sum_channel = np.zeros(3)
        squared_sum_channel = np.zeros(3)
        total_pixels = 0
        total_images = len(self.files)

        # Iterate through all images to accumulate sums
        for file in self.files:
            img = Image.open(file)

            img_array = np.array(img)
            total_pixels += img_array.size / 3
            sum_channel += np.sum(img_array, axis=(0, 1)) / 255.0
            squared_sum_channel += np.sum((img_array / 255.0) ** 2, axis=(0, 1))

        means = sum_channel / total_pixels
        stds = np.sqrt((squared_sum_channel / total_pixels) - (means ** 2))

        return means.tolist(), stds.tolist()

    def calculate_mean_std_gray(self):
        sum_channel = np.zeros(1)  # Grayscale has only one channel
        squared_sum_channel = np.zeros(1)
        total_pixels = 0
        total_images = len(self.files)

        # Iterate through all images to accumulate sums
        for file in self.files:
            img = Image.open(file).convert("L")  # Convert image to grayscale

            img_array = np.array(img)
            total_pixels += img_array.size
            sum_channel += np.sum(img_array) / 255.0
            squared_sum_channel += np.sum((img_array / 255.0) ** 2)

        means = sum_channel / total_pixels
        stds = np.sqrt((squared_sum_channel / total_pixels) - (means ** 2))

        return [means], [stds]

    def get_transform(self):
        if self.gray_scale:
            return transforms.Compose([
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=torch.rand(1).item(),
                        contrast=torch.rand(1).item(),
                        saturation=torch.rand(1).item(),
                        hue=torch.rand(1).item() * 0.5,
                    )
                ]),
                transforms.Resize((120, 120)),
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds)
            ])
        else:
            return transforms.Compose([
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=torch.rand(1).item(),
                        contrast=torch.rand(1).item(),
                        saturation=torch.rand(1).item(),
                        hue=torch.rand(1).item() * 0.5,
                    )
                ]),
                transforms.Resize((120, 120)),
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds)
            ])

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.files)

    def __getitem__(self, index):
        """Returns the index-th data item of the dataset."""
        if index < 0 or index >= self.__len__():
            raise IndexError(
                'Wrong index (must be in [{}, {}], given: {})'
                .format(0, self.__len__(), index)
            )

        # Load image and transform
        img = Image.open(self.files[index])
        img = self.transform(img)

        # Get label
        class_name = os.path.basename(os.path.dirname(self.files[index]))
        label = self.classes.index(class_name)

        return img, torch.tensor(label)