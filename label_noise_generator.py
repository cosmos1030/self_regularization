import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

# Define the data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

def introduce_label_noise(dataset, noise_ratio=0.7, save_path="noisy_labels.npy", reload=False, seed=42):
    """
    Introduces label noise with a fixed seed for reproducibility.

    Args:
        dataset (torchvision.datasets.CIFAR10): Original dataset.
        noise_ratio (float): Percentage of labels to randomize (0.7 means 70% noisy labels).
        save_path (str): File path to save/load noisy labels.
        reload (bool): If True, reloads noisy labels from file instead of generating new ones.
        seed (int): Random seed for reproducibility.

    Returns:
        noisy_targets (list): List of modified labels with noise introduced.
    """
    if reload and os.path.exists(save_path):
        print(f"Loading noisy labels from {save_path}")
        return np.load(save_path).tolist()

    # Set the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    num_samples = len(dataset.targets)
    num_noisy = int(num_samples * noise_ratio)  # Number of samples to corrupt

    indices = list(range(num_samples))
    random.shuffle(indices)  # Shuffle indices to select which labels to change
    noisy_indices = indices[:num_noisy]  # Select first num_noisy indices for corruption

    noisy_targets = dataset.targets.copy()  # Copy original labels
    all_classes = list(range(10))  # CIFAR-10 has 10 classes

    for i in noisy_indices:
        original_label = noisy_targets[i]
        possible_labels = [label for label in all_classes if label != original_label]  # Avoid same label
        noisy_targets[i] = random.choice(possible_labels)  # Assign a random incorrect label

    # Save noisy labels for future use
    np.save(save_path, np.array(noisy_targets))
    print(f"Noisy labels saved to {save_path}")

    return noisy_targets

# Choose noise level (e.g., 70% noise)
noise_ratio = 0.3
seed = 5  # Set seed for reproducibility
save_path = f"./data/noisy_labels{int(noise_ratio * 100)}_seed{int(seed)}.npy"

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Introduce label noise (either generate or reload from saved file)
trainset.targets = introduce_label_noise(trainset, noise_ratio=noise_ratio, save_path=save_path, reload=True, seed=seed)

print(f"Label noise applied with {noise_ratio * 100}% noise with noise seed {seed}. Noisy dataset ready!")
