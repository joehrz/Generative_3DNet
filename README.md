# Generative 3DWheatNet (BI-Net)

> **Note**: This project is still a **work in progress**.

Implementation of the BI-Net paper:  
[**3D Point Cloud Shape Generation with Collaborative Learning of Generative Adversarial Network and Auto-Encoder**](https://www.mdpi.com/2072-4292/16/10/1772)

**BI-Net** (Bidirectional Network) is a **collaborative** Auto-Encoder (AE) + Generative Adversarial Network (GAN) designed for **3D point-cloud** data. Leveraging **TreeGCN** expansions, BI-Net can **reconstruct** real point clouds (forward AE direction) and **generate** new shapes (reverse GAN direction). This approach is particularly well-suited to limited 3D datasets (e.g., wheat plants), using the AE to guide the GAN’s discriminator and generator.

---

## Features

1. **Forward (AE) Direction**  
   - Encodes real point clouds \(\mathbf{X}\) into a latent code \(\mathbf{z}\).  
   - Reconstructs \(\mathbf{X}\) via TreeGCN-based decoding.  
   - Supports Chamfer/EMD loss for high‐fidelity reconstruction.

2. **Reverse (GAN) Direction**  
   - Generates point clouds from random noise, discriminated by a shared “En–Di” module.  
   - Implements **Wasserstein GAN** with Gradient Penalty (WGAN-GP) for stable training.  
   - Includes optional **NNME** (Nearest Neighbor Mutual Exclusion) loss to ensure uniform point distribution.

3. **TreeGCN Expansion**  
   - Hierarchical branching from a small latent code to a large set of 3D points.  
   - Flexible configuration (layers/degrees) for different shape complexities.

4. **Data Preprocessing**  
   - Voxel‐based downsampling and/or farthest point sampling for consistent point counts.  
   - Normalization to a unit sphere, plus optional random noise augmentation.

5. **Multi‐Notebook Workflow**  
   - Explore data, model architectures, training procedures, and evaluation routines in separate Jupyter notebooks (if provided).

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