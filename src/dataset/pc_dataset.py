# File: src/dataset/pc_dataset.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    """
    A dataset class for point-cloud data (optionally with RGB).
    Expects data files directly under the `root` directory:
        root/*.npy, root/*.npz, root/*.pt, root/*.txt, etc.
    
    - Supports files: .npy, .npz, .pt, .txt
    - Typically expects arrays of shape (N, 3) for XYZ or (N, 6) for XYZ+RGB.
    - If using .npz, it looks for a 'points' key by default (or else defaults to the
      first array found in the file).
    - An optional transform can be applied to the loaded NumPy array before
      it is converted to a PyTorch tensor.
    """

    def __init__(self, root='data/processed', transform=None):
        """
        Args:
            root (str): Directory containing the data files (e.g. 'data/processed/').
            transform (callable, optional): A function that takes a NumPy array
                                            and returns a modified NumPy array
                                            (could be augmentations, scaling, etc.).
        """
        super().__init__()
        self.root = root
        self.transform = transform

        # Check that the directory exists
        if not os.path.isdir(self.root):
            raise ValueError(f"Provided root directory does not exist: {self.root}")

        # Gather valid files
        self.files = sorted(
            f
            for f in os.listdir(self.root)
            if f.endswith('.npy') or f.endswith('.npz')
               or f.endswith('.pt')  or f.endswith('.txt')
        )

        if len(self.files) == 0:
            raise ValueError(f"No valid point-cloud files found in directory: {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Build the full path to the file
        file_name = self.files[idx]
        file_path = os.path.join(self.root, file_name)

        # Load the file into a NumPy array
        if file_path.endswith('.npy'):
            pc = np.load(file_path)
        elif file_path.endswith('.npz'):
            data = np.load(file_path)
            # If there's a 'points' key, use it; otherwise fall back to the first array
            if 'points' in data:
                pc = data['points']
            else:
                arr_keys = list(data.keys())
                if len(arr_keys) == 0:
                    raise ValueError(f"No arrays found in {file_path}")
                pc = data[arr_keys[0]]
        elif file_path.endswith('.pt'):
            # If .pt stored a Torch tensor or a NumPy-like structure
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

        # Apply optional transform/augmentation
        if self.transform is not None:
            pc = self.transform(pc)

        # Convert to torch.Tensor
        return torch.from_numpy(pc)
