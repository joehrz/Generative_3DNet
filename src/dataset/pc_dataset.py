
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class WheatPointCloudDataset(Dataset):
    """
    Example dataset class for wheat point clouds.
    - Expects data in 'data/processed/' or user-specified root.
    - Could load .npy, .txt, .pt, etc. containing (N, 3) points per sample.
    """
    def __init__(self, root='data/processed', split='train', transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        # Example: we assume there's a folder for each split
        # e.g.: data/processed/train/*.npy
        self.file_dir = os.path.join(self.root, split)
        self.files = sorted([f for f in os.listdir(self.file_dir) 
                             if f.endswith('.npy') or f.endswith('.pt') or f.endswith('.txt')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.file_dir, self.files[idx])

        # Load the point cloud
        if file_path.endswith('.npy'):
            pc = np.load(file_path)  # shape (N, 3)
        elif file_path.endswith('.pt'):
            pc = torch.load(file_path).numpy()  # or handle differently
        elif file_path.endswith('.txt'):
            pc = np.loadtxt(file_path)
        else:
            raise ValueError("Unsupported file format.")

        # Convert to float32
        pc = pc.astype(np.float32)

        # Optional transform/augmentation
        if self.transform is not None:
            pc = self.transform(pc)

        return torch.from_numpy(pc)  # (N, 3), type: torch.float32