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
    - An optional transform can be applied to the loaded data. It is applied after
      the data is converted to a PyTorch tensor.
    """

    def __init__(self, root='data/processed', transform=None):
        """
        Args:
            root (str): Directory containing the data files (e.g. 'data/processed/').
            transform (callable, optional): A function that takes a PyTorch tensor
                                            and returns a modified tensor.
        """
        super().__init__()
        self.root = root
        self.transform = transform

        if not os.path.isdir(self.root):
            raise ValueError(f"Provided root directory does not exist: {self.root}")

        self.files = sorted(
            f
            for f in os.listdir(self.root)
            if f.endswith(('.npy', '.npz', '.pt', '.txt'))
        )

        if not self.files:
            raise ValueError(f"No valid point-cloud files found in directory: {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.files[idx])
        pc_np = None

        try:
            if file_path.endswith('.npy'):
                pc_np = np.load(file_path)
            elif file_path.endswith('.npz'):
                data = np.load(file_path)
                if 'points' in data:
                    pc_np = data['points']
                else:
                    pc_np = data[list(data.keys())[0]]
            elif file_path.endswith('.pt'):
                tensor_data = torch.load(file_path)
                pc_np = tensor_data.cpu().numpy() if isinstance(tensor_data, torch.Tensor) else np.array(tensor_data)
            elif file_path.endswith('.txt'):
                pc_np = np.loadtxt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            pc_tensor = torch.from_numpy(pc_np.astype(np.float32))

            if self.transform:
                pc_tensor = self.transform(pc_tensor)

            return pc_tensor

        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raise e
