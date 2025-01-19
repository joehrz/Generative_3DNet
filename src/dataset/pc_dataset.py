# File: src/dataset/pc_dataset.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    """
    A dataset class for point-cloud data (optionally with RGB).
    Assumes data is stored in `root/split` subfolders:
      e.g. data/processed/train/*.npy or .npz, etc.
    
    - Supports files: .npy, .npz, .pt, .txt
    - Typically expects arrays of shape (N, 3) for XYZ or (N, 6) for XYZ+RGB.
    - If using .npz, it looks for a 'points' key by default (or the first array if absent).
    - Applies an optional transform, then returns a torch.Tensor (float32) of shape (N, D)
      where D=3 for XYZ or D=6 for XYZRGB, etc.
    """

    def __init__(self, root='data/processed', split='train', transform=None):
        """
        Args:
            root (str): Base directory, e.g. 'data/processed'
            split (str): Subfolder name, e.g. 'train', 'val', 'test'
            transform (callable, optional): A function that takes a NumPy array
                                            and returns a modified NumPy array.
                                            (Could be augmentations, scaling, etc.)
        """
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        # Path to the folder containing files for this split
        self.file_dir = os.path.join(self.root, self.split)
        
        # Gather valid files
        self.files = sorted([
            f for f in os.listdir(self.file_dir)
            if f.endswith('.npy') or f.endswith('.npz') or f.endswith('.pt') or f.endswith('.txt')
        ])

    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, idx):
        file_path = os.path.join(self.file_dir, self.files[idx])

        # Load the file into a NumPy array
        if file_path.endswith('.npy'):
            pc = np.load(file_path)  # e.g. shape (N, 3) or (N, 6)
        elif file_path.endswith('.npz'):
            data = np.load(file_path)
            # If there's a known key 'points', use it; else fallback to first array
            if 'points' in data:
                pc = data['points']
            else:
                # fallback: grab first array from .npz if no 'points' key
                arr_keys = list(data.keys())
                if len(arr_keys) == 0:
                    raise ValueError(f"No arrays found in {file_path}")
                pc = data[arr_keys[0]]
        elif file_path.endswith('.pt'):
            # If .pt stored a NumPy or Torch tensor, we handle it
            tensor_data = torch.load(file_path)
            if isinstance(tensor_data, torch.Tensor):
                pc = tensor_data.cpu().numpy()
            else:
                pc = np.array(tensor_data, dtype=np.float32)
        elif file_path.endswith('.txt'):
            pc = np.loadtxt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Convert to float32 array
        pc = pc.astype(np.float32)

        # Optional transform/augmentation (could be random rotations, etc.)
        if self.transform is not None:
            pc = self.transform(pc)  # still a NumPy array

        # Convert to torch Tensor of shape (N, D), float32
        return torch.from_numpy(pc)