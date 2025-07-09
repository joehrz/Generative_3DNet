# File: src/dataset/pc_dataset.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from src.utils.validation import validate_point_cloud_tensor, validate_file_path, ValidationError
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

        # Validate root directory
        try:
            validate_file_path(self.root, "root directory", check_exists=True)
        except ValidationError as e:
            raise ValueError(str(e))

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
                tensor_data = torch.load(file_path, map_location='cpu', weights_only=True)
                pc_np = tensor_data.cpu().numpy() if isinstance(tensor_data, torch.Tensor) else np.array(tensor_data)
            elif file_path.endswith('.txt'):
                pc_np = np.loadtxt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            pc_tensor = torch.from_numpy(pc_np.astype(np.float32))
            
            # Validate the loaded point cloud
            try:
                pc_tensor = validate_point_cloud_tensor(
                    pc_tensor,
                    name=f"point_cloud_{file_path}",
                    min_points=10,
                    max_points=50000,
                    allow_batch=False
                )
            except ValidationError as e:
                raise ValueError(f"Invalid point cloud in {file_path}: {e}")

            if self.transform:
                pc_tensor = self.transform(pc_tensor)

            return pc_tensor

        except FileNotFoundError:
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")
        except np.load as e:
            raise ValueError(f"Error loading numpy file {file_path}: {e}")
        except torch.jit.RecursiveScriptModule as e:
            raise ValueError(f"Error loading PyTorch file {file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading file {file_path}: {e}")
