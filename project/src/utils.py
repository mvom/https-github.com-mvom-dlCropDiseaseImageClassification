# -*- coding: utf-8 -*-
"""utils.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1U49jew1FW3Y7RSGplKBVlIBw9wFc7N-x
"""

import os
import shutil
import subprocess
import random
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader
from PIL import ImageFile

# Set a fixed seed for reproducibility across various libraries
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
ImageFile.LOAD_TRUNCATED_IMAGES = True

def extract_folder(repo_url, clone_dir, folder_name):
    """
    Extracts a specific folder from a cloned repository.

    Args:
        repo_url (str): URL of the repository to clone.
        clone_dir (str): Directory where the repository is cloned.
        folder_name (str): Name of the folder to extract from the cloned repository.

    Returns:
        dict: A dictionary where keys are class names and values are the number of files in each class.
    """
    if os.path.exists(folder_name):
        os.chdir("..")
        shutil.move(os.path.join(clone_dir, folder_name), folder_name)
        print(f"Folder '{folder_name}' extracted successfully.")

        class_counts = {}
        total_files = 0

        # Count number of files in each directory
        for root, dirs, files in os.walk(folder_name):
            for directory in dirs:
                dir_path = os.path.join(root, directory)
                num_files = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
                class_counts[directory] = num_files
                total_files += num_files
                print(f"Directory '{directory}' contains {num_files} files.")

        print(f"Total number of files: {total_files}")

    else:
        print(f"Folder '{folder_name}' not found in the repository.")
        class_counts = {}

    print()
    print(f"Total number of classes : '{len(class_counts)}'.")
    print(class_counts)

    return class_counts

def clone_repo(repo_url, clone_dir):
    """
    Clones a repository from a given URL into a specified directory.

    Args:
        repo_url (str): URL of the repository to clone.
        clone_dir (str): Directory where the repository will be cloned.

    Returns:
        None
    """
    subprocess.run(["git", "clone", repo_url, clone_dir])
    os.chdir(clone_dir)

def show_images(img):
    """
    Displays an image using matplotlib.

    Args:
        img (Tensor): A PyTorch tensor representing the image.

    Returns:
        None
    """
    plt.imshow(transforms.functional.to_pil_image(img))
    plt.show()

def plot_class_histogram(class_counts):
    """
    Plot a histogram of the distribution of the number of samples for each class of crop disease.

    Args:
        class_counts (dict): A dictionary where keys are class names and values are the number of samples for each class.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Extract class names and counts
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # Plot histogram
    sns.barplot(x=counts, y=classes, hue=classes, dodge=False, palette='husl', legend=False)
    for i, count in enumerate(counts):
        plt.text(count + 20, i, str(count), va='center')

    plt.xlabel('Number of samples')
    plt.ylabel('Crop disease classes')
    plt.title('Histogram of the distribution of the number of samples for each class of crop disease')
    plt.tight_layout()
    plt.show()

def plot_training_results(train_avg_loss, validation_avg_loss, validation_accuracy, is_validation=True):
    """
    Defines a method to plot training results.

    Args:
        train_avg_loss (list): List of average training losses per epoch.
        validation_avg_loss (list): List of average validation/test losses per epoch.
        validation_accuracy (list): List of validation/test accuracies per epoch.
        is_validation (bool): Indicates whether the data is for validation or test.
    """
    sns.set(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    if is_validation:
        loss_label = 'Validation Loss'
        accuracy_label = 'Validation Accuracy'
    else:
        loss_label = 'Test Loss'
        accuracy_label = 'Test Accuracy'

    # Plot train and validation/test loss
    ax1.plot(train_avg_loss, label='Train Loss')
    ax1.plot(validation_avg_loss, label=loss_label)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')

    # Plot validation/test accuracy
    ax2.plot(validation_accuracy, label=accuracy_label)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='upper left')

    plt.show()