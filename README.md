# Generative 3DWheatNet (BI-Net)

> **Note**: This project is still a **work in progress**.

This is an implementation of the BI-Net paper:  
[**3D Point Cloud Shape Generation with Collaborative Learning of Generative Adversarial Network and Auto-Encoder**](https://www.mdpi.com/2072-4292/16/10/1772)

**BI-Net** (Bidirectional Network) is a **collaborative** Auto-Encoder (AE) and Generative Adversarial Network (GAN) for **3D point-cloud** data. Using **TreeGCN** expansions, BI-Net can **reconstruct** real point clouds in one direction and **generate** new shapes in the other direction. This approach is particularly suited for limited 3D datasets (like wheat plants), leveraging an AE to guide the GAN’s discriminator and generator.

---

## Features

- **Forward (AE) Direction**  
  - Encodes real point clouds into a latent code and reconstructs them via TreeGCN-based decoding.
  - Supports Chamfer/EMD loss for reconstruction fidelity.

- **Reverse (GAN) Direction**  
  - Generates point clouds from random noise, discriminated by the shared “En–Di” module.
  - Implements Wasserstein GAN with Gradient Penalty (WGAN-GP) for stable training.
  - Optional **NNME** (Nearest Neighbor Mutual Exclusion) loss for uniform point distribution.

- **TreeGCN Expansion**  
  - Hierarchical branching from a small latent code to a large set of 3D points.
  - Flexible degrees / layers for different shape complexity.

- **Data Preprocessing**  
  - Voxel-based downsampling, farthest point sampling to unify point counts, normalization to a unit sphere.

- **Multiple Notebooks**  
  - Exploration of data, model architecture, training, and evaluation routines.

---

1. **Clone the Repository:**


   ```bash
    git clone https://github.com/YourUsername/Generative_3DWheatNet.git
    cd Generative_3DWheatNet

2. **Build Instructions**
    pip install -r requirements.txt


**Data Preparation**

    Place raw .ply (or other 3D formats supported by Open3D) point-cloud files in data/raw/.
    The folder data/processed/ (or data/preprocessed) will store generated .npz files after running preprocessing steps.

## Usage

**Data Preprocessing**

To downsample, unify point counts, and normalize:

python src/main.py --preprocess \
    --input_dir data/raw \
    --output_dir data/processed \
    --voxel_size 0.02 \
    --num_points 2048 \
    --use_fps


**Training**

Once you have preprocessed .npz files in data/processed, you can train BI-Net:

python src/main.py --train \
    --data_dir data/processed \
    --batch_size 8 \
    --epochs 10 \
    --device cuda

Adjust parameters (batch size, epochs, GPU device, etc.) as needed. You can also specify other flags like learning rates or latent dimensions if supported in main.py.

**Evaluation**

python scripts/evaluate.py \
    --model_checkpoint bi_net_checkpoint.pth \
    --batch_size 8 \
    --latent_dim 96 \
    --data_root data/processed \
    --split test