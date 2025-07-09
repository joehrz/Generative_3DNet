# Small Dataset Training Guide for BI-Net

This guide provides comprehensive strategies for training the BI-Net 3D point cloud generation model with small datasets. The implementations below address common challenges in small dataset scenarios.

## Table of Contents
1. [Overview](#overview)
2. [Enhanced Data Augmentation](#enhanced-data-augmentation)
3. [Training Configuration](#training-configuration)
4. [Regularization Techniques](#regularization-techniques)
5. [Cross-Validation](#cross-validation)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)

## Overview

Small dataset training poses unique challenges:
- **Overfitting**: Model memorizes training data
- **Limited diversity**: Insufficient variation in training examples
- **Unstable training**: GANs are particularly sensitive to small datasets
- **Poor generalization**: Model fails on unseen data

Our enhanced pipeline addresses these issues through:
- **Advanced augmentations**: 12+ geometric and semantic augmentations
- **Extended training**: 100 epochs with progressive phases
- **Regularization**: Dropout, weight decay, spectral normalization
- **Cross-validation**: K-fold validation for robust evaluation

## Enhanced Data Augmentation

### Basic Augmentations (Already Available)
```yaml
# Standard augmentations
augment_rotate: True              # Z-axis rotation
augment_flip: True                # X/Y axis flipping
augment_scale: True               # Uniform scaling (0.9-1.1)
augment_noise_std: 0.005         # Gaussian noise
augment_jitter_sigma: 0.01       # Point jittering
```

### Advanced Augmentations (New)
```yaml
# Advanced geometric augmentations
augment_perspective: True
augment_perspective_strength: 0.1

augment_elastic_deformation: True
augment_elastic_strength: 0.05
augment_elastic_control_points: 4

augment_occlusion: True
augment_occlusion_ratio: 0.1

augment_dropout: True
augment_dropout_ratio: 0.05

augment_rotate_3d: True
augment_rotate_3d_max_angle: 0.3

augment_point_resampling: True
augment_resampling_ratio: 0.1
```

### Augmentation Details

1. **Perspective Transform**: Simulates perspective distortion
2. **Elastic Deformation**: Non-rigid deformation using radial basis functions
3. **Occlusion**: Random point removal and padding
4. **Dropout**: Replace points with noise
5. **3D Rotation**: Full 3D rotation (not just Z-axis)
6. **Point Resampling**: Interpolate new points from neighbors

## Training Configuration

### Extended Training Schedule
```yaml
training:
  epochs: 100                     # Increased from 10
  warmup_epochs: 20              # Increased from 10
  batch_size: 12                 # Keep small for stability
  
  # Progressive training phases
  progressive_training: True
  progressive_phases: [30, 30, 40]  # Total: 100 epochs
  
  # Advanced scheduling
  use_cosine_annealing: True
  cosine_annealing_T_max: 100
  cosine_annealing_eta_min: 0.00001
```

### Learning Rate Strategy
- **Phase 1 (0-30 epochs)**: Autoencoder pre-training + basic GAN
- **Phase 2 (30-60 epochs)**: Full adversarial training
- **Phase 3 (60-100 epochs)**: Fine-tuning with reduced learning rate

## Regularization Techniques

### Model Regularization
```yaml
# Regularization parameters
weight_decay: 0.0001           # L2 regularization
dropout_rate: 0.1              # Dropout in FC layers
use_spectral_norm: True        # Spectral normalization for stability
gradient_clip_norm: 1.0        # Gradient clipping
```

### Implementation Details
1. **Dropout**: Applied in encoder/discriminator FC layers
2. **Spectral Normalization**: Constrains Lipschitz constant
3. **Weight Decay**: L2 penalty on model parameters
4. **Gradient Clipping**: Prevents exploding gradients

## Cross-Validation

### Configuration
```yaml
# Cross-validation for small datasets
use_cross_validation: True     # Enable k-fold CV
cv_folds: 5                   # Number of folds
cv_random_state: 42           # Reproducibility
ensemble_predictions: True     # Use model ensemble
```

### Benefits
- **Robust evaluation**: Better estimate of model performance
- **Model selection**: Choose best fold or ensemble
- **Reduced overfitting**: Multiple train/val splits
- **Ensemble predictions**: Average multiple models

## Usage Examples

### 1. Standard Small Dataset Training
```bash
# Update configuration for small dataset
# Set epochs: 100, enable advanced augmentations, add regularization

# Run training
python main.py --config src/configs/default_config.yaml --train
```

### 2. Cross-Validation Training
```bash
# Enable cross-validation in config
# Set use_cross_validation: True

# Run cross-validation
python main.py --config src/configs/default_config.yaml --cross-validate
```

### 3. Complete Pipeline for Small Dataset
```bash
# 1. Preprocess data
python main.py --config src/configs/default_config.yaml --preprocess

# 2. Run cross-validation (recommended for small datasets)
python main.py --config src/configs/default_config.yaml --cross-validate

# 3. Train final model (optional, if not using CV)
python main.py --config src/configs/default_config.yaml --train

# 4. Evaluate
python main.py --config src/configs/default_config.yaml --eval --checkpoint best_cv_model.pth
```

## Best Practices

### 1. Data Preparation
- **Quality over quantity**: Ensure high-quality, diverse samples
- **Balanced dataset**: Avoid class imbalance if applicable
- **Proper preprocessing**: Consistent normalization and point counts

### 2. Training Strategy
- **Start with cross-validation**: Get robust performance estimates
- **Monitor overfitting**: Watch train vs. validation metrics
- **Early stopping**: Stop if validation loss plateaus
- **Save checkpoints**: Regular model saving for recovery

### 3. Hyperparameter Tuning
- **Conservative learning rates**: Start with 0.0002, reduce if unstable
- **Batch size**: Keep small (8-16) for small datasets
- **Augmentation strength**: Start conservative, increase gradually
- **Regularization**: Increase dropout/weight_decay if overfitting

### 4. Evaluation
- **Multiple metrics**: EMD, Chamfer distance, visual inspection
- **Cross-validation**: Use CV for reliable performance estimates
- **Ensemble methods**: Combine multiple models for better results
- **Qualitative analysis**: Manually inspect generated samples

### 5. Troubleshooting Small Dataset Issues

#### Symptom: Rapid overfitting
**Solutions:**
- Increase dropout rate (0.2-0.3)
- Add more aggressive augmentations
- Reduce model capacity
- Use stronger regularization

#### Symptom: Training instability
**Solutions:**
- Enable spectral normalization
- Reduce learning rates
- Use gradient clipping
- Increase warmup epochs

#### Symptom: Poor generation quality
**Solutions:**
- Increase training epochs
- Use ensemble predictions
- Adjust GAN loss weights
- Add more diverse augmentations

#### Symptom: Mode collapse
**Solutions:**
- Use progressive training
- Adjust discriminator/generator balance
- Increase batch diversity through augmentation
- Use different random seeds

## Advanced Techniques (Future Work)

### Transfer Learning
- Pre-train on ShapeNet dataset
- Fine-tune on target small dataset
- Feature extraction from pre-trained models

### Self-Supervised Learning
- Contrastive learning on point clouds
- Point cloud completion tasks
- Rotation prediction tasks

### Data Generation
- Use trained model to generate synthetic training data
- Progressive data augmentation
- GAN-based data synthesis

## Configuration Template for Small Datasets

```yaml
# Optimized configuration for small datasets
training:
  epochs: 100
  warmup_epochs: 20
  batch_size: 12
  
  # Learning rates
  lr_enc: 0.0001      # Reduced for stability
  lr_dec: 0.0001
  lr_disc: 0.0001
  
  # Regularization
  weight_decay: 0.0001
  dropout_rate: 0.15   # Increased for small datasets
  use_spectral_norm: True
  gradient_clip_norm: 1.0
  
  # Advanced augmentations (all enabled)
  augment_perspective: True
  augment_elastic_deformation: True
  augment_occlusion: True
  augment_dropout: True
  augment_rotate_3d: True
  augment_point_resampling: True
  
  # Cross-validation
  use_cross_validation: True
  cv_folds: 5
  ensemble_predictions: True

# Model configuration
model:
  lambda_gp: 10.0     # Gradient penalty weight
  lambda_nnme: 1.0    # Uniformity loss weight
```

This comprehensive approach should significantly improve training performance on small datasets while maintaining generation quality.