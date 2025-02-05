# Generative 3DWheatNet (BI-Net)
> **Note**: This project is still a **work in progress**.

This is an implementation of the BI-Net paper for educational purposes:  
[**3D Point Cloud Shape Generation with Collaborative Learning of Generative Adversarial Network and Auto-Encoder**](https://www.mdpi.com/2072-4292/16/10/1772)

**BI-Net** (Bidirectional Network) is a **collaborative** Auto-Encoder (AE) and Generative Adversarial Network (GAN) for **3D point-cloud** data. Using **TreeGCN** expansions, BI-Net can **reconstruct** real point clouds in one direction and **generate** new shapes in the other direction. 

---

## Features

- **Forward (AE) Direction**  
  - Encodes real point clouds into a latent code and reconstructs them via a TreeGCN-based decoder.
  - Supports Chamfer / EMD loss for high-fidelity reconstruction.

- **Reverse (GAN) Direction**  
  - Generates point clouds from random noise, using the shared “En–Di” module as a discriminator.
  - Implements Wasserstein GAN with Gradient Penalty (WGAN-GP) for stable training.
  - Optional **NNME** (Nearest Neighbor Mutual Exclusion) loss encourages uniform point distributions.

- **TreeGCN Expansion**  
  - Hierarchical branching from a small latent code to a large 3D point set.
  - Flexible degrees / layers to adapt shape complexity.

- **Data Preprocessing**  
  - Voxel-based downsampling, farthest point sampling, and normalization to a unit sphere.

- **Multiple Notebooks**  
  - Exploration of data, model architectures, training procedures, and evaluation.

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourUsername/Generative_3DWheatNet.git
   cd Generative_3DWheatNet
   ```

2. **Build Instructions**:
   ```bash
   pip install -r requirements.txt
   ```

**Data Preparation**

Place raw .ply (or other 3D formats supported by Open3D) point-cloud files in `data/raw/`.  
The folder `data/processed/` (or `data/preprocessed`) will store generated .npz files after running preprocessing steps.

## Usage

**Data Preprocessing**

To downsample, unify point counts, and normalize:
```bash
python src/main.py --preprocess \
    --input_dir data/raw \
    --output_dir data/processed \
    --voxel_size 0.02 \
    --num_points 2048 \
    --use_fps
```

**Training**

Once you have preprocessed .npz files in `data/processed`, you can train BI-Net:
```bash
python src/main.py --train \
    --data_dir data/processed \
    --batch_size 8 \
    --epochs 10 \
    --device cuda
```
Adjust parameters (batch size, epochs, GPU device, etc.) as needed. You can also specify other flags like learning rates or latent dimensions if supported in `main.py`.

**Evaluation**
```bash
python scripts/evaluate.py \
    --model_checkpoint bi_net_checkpoint.pth \
    --batch_size 8 \
    --latent_dim 96 \
    --data_root data/processed \
    --split test
```

## Things to Do Next
- **Model Improvement**  
  Investigate and address the reasons for sub-par model performance. Consider hyperparameter tuning, architecture adjustments, or additional training data.
- **Add Data Augmentations**  
  Implement rotations, flips, or random noise injections specifically tailored to wheat data, improving model robustness.
- **Tune TreeGCN Structure**  
  Experiment with different degrees, deeper or shallower expansions, or alternate activation functions.
- **Refine NNME Loss**  
  Investigate advanced metrics for point uniformity or coverage. Try weighting schemes or multi-scale nearest neighbors.
- **Integrate Additional Losses**  
  Explore other 3D losses (e.g., EMD loss) if relevant.
- **Benchmark on Baseline Datasets**  
  Validate on other point-cloud datasets (e.g., ShapeNet or agricultural data) to confirm generalization.
- **Improve Logging/Visualization**  
  Enhance real-time monitoring, add 3D shape visualizations, or integrate with TensorBoard to compare models.
