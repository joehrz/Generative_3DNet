## src/data_processing/dataset_splitting.py

'''Split the dataset into a train, val, and test set'''

import os
import shutil
import random
from typing import Tuple


def split_dataset(input_dir: str, output_dir: str, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42) -> None:
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        input_dir (str): Directory containing the data to split.
        output_dir (str): Directory to save the splits.
        split_ratios (Tuple[float, float, float]): Ratios for train, val, and test splits.
        seed (int): Random seed for reproducibility.
    """
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1."
    random.seed(seed)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Gather all .npz files
    files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    if not files:
        print(f"No .npz found in {input_dir}")
        return
    
    random.shuffle(files)

    train_end = int(split_ratios[0] * len(files))
    val_end = train_end + int(split_ratios[1] * len(files))
    
    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    for split_name, split_files in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for filename in split_files:
            shutil.copy(os.path.join(input_dir, filename), os.path.join(split_dir, filename))
        print(f"Copied {len(split_files)} files to {split_dir}")


