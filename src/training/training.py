# File: src/training/training.py

"""
This module contains the main high-level pipeline for orchestrating the
training, evaluation, and generation processes.
"""

import os
import torch
import numpy as np
import datetime
from torch.utils.data import DataLoader

from src.dataset.pc_dataset import PointCloudDataset
from src.models.bi_net import BiNet
from src.utils.train_utils import train_binet
from src.utils.losses import evaluate_on_loader_emd_chamfer
from src.utils.pc_utils import (random_rotate, random_flip, random_noise, random_scale, jitter_points, 
                               random_perspective_transform, random_elastic_deformation, random_occlusion, 
                               random_dropout, random_rotate_3d, random_point_resampling, PointCloudAugmentation)


# --- Augmentation Wrapper Classes (to fix Windows multiprocessing pickling issues) ---
# These classes wrap the augmentation functions to make them easily picklable by the DataLoader.

class RandomFlipTransform:
    def __init__(self, p_x=0.5, p_y=0.5, p_z=0.0):
        self.p_x, self.p_y, self.p_z = p_x, p_y, p_z
    def __call__(self, pc):
        return random_flip(pc, self.p_x, self.p_y, self.p_z)

class RandomNoiseTransform:
    def __init__(self, std=0.01):
        self.std = std
    def __call__(self, pc):
        return random_noise(pc, std=self.std)

class RandomScaleTransform:
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.min_scale, self.max_scale = min_scale, max_scale
    def __call__(self, pc):
        return random_scale(pc, self.min_scale, self.max_scale)

class JitterPointsTransform:
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma, self.clip = sigma, clip
    def __call__(self, pc):
        return jitter_points(pc, self.sigma, self.clip)

class PerspectiveTransform:
    def __init__(self, strength=0.1):
        self.strength = strength
    def __call__(self, pc):
        return random_perspective_transform(pc, self.strength)

class ElasticDeformationTransform:
    def __init__(self, strength=0.1, num_control_points=4):
        self.strength = strength
        self.num_control_points = num_control_points
    def __call__(self, pc):
        return random_elastic_deformation(pc, self.strength, self.num_control_points)

class OcclusionTransform:
    def __init__(self, occlusion_ratio=0.1):
        self.occlusion_ratio = occlusion_ratio
    def __call__(self, pc):
        return random_occlusion(pc, self.occlusion_ratio)

class DropoutTransform:
    def __init__(self, dropout_ratio=0.1):
        self.dropout_ratio = dropout_ratio
    def __call__(self, pc):
        return random_dropout(pc, self.dropout_ratio)

class Rotate3DTransform:
    def __init__(self, max_angle=0.5):
        self.max_angle = max_angle
    def __call__(self, pc):
        return random_rotate_3d(pc, self.max_angle)

class PointResamplingTransform:
    def __init__(self, resample_ratio=0.2):
        self.resample_ratio = resample_ratio
    def __call__(self, pc):
        return random_point_resampling(pc, self.resample_ratio)


def run_training_pipeline(config, logger, do_train=False, do_eval=False, do_generate=False, ckpt_name=None):
    """
    Coordinates the main ML pipeline based on command-line flags.
    """
    # --- 1. Device Configuration ---
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"[INFO] Using device: {device}")
    config.training.device = device

    # --- 2. Datasets & DataLoaders ---
    logger.info("[INFO] Setting up datasets and augmentations...")
    train_transforms_list = []
    
    if config.training.augment_rotate:
        train_transforms_list.append(random_rotate) # This function is already defined at the top level and is picklable.
    if config.training.augment_flip:
        train_transforms_list.append(RandomFlipTransform(p_x=0.5, p_y=0.5, p_z=0.0))
    if config.training.augment_noise_std > 0:
        train_transforms_list.append(RandomNoiseTransform(std=config.training.augment_noise_std))
    if config.training.augment_scale:
        train_transforms_list.append(RandomScaleTransform(min_scale=config.training.augment_min_scale, max_scale=config.training.augment_max_scale))
    if config.training.augment_jitter_sigma > 0:
        train_transforms_list.append(JitterPointsTransform(sigma=config.training.augment_jitter_sigma, clip=config.training.augment_jitter_clip))
    
    # --- Advanced Augmentations ---
    if getattr(config.training, 'augment_perspective', False):
        train_transforms_list.append(PerspectiveTransform(strength=config.training.augment_perspective_strength))
    if getattr(config.training, 'augment_elastic_deformation', False):
        train_transforms_list.append(ElasticDeformationTransform(
            strength=config.training.augment_elastic_strength,
            num_control_points=config.training.augment_elastic_control_points
        ))
    if getattr(config.training, 'augment_occlusion', False):
        train_transforms_list.append(OcclusionTransform(occlusion_ratio=config.training.augment_occlusion_ratio))
    if getattr(config.training, 'augment_dropout', False):
        train_transforms_list.append(DropoutTransform(dropout_ratio=config.training.augment_dropout_ratio))
    if getattr(config.training, 'augment_rotate_3d', False):
        train_transforms_list.append(Rotate3DTransform(max_angle=config.training.augment_rotate_3d_max_angle))
    if getattr(config.training, 'augment_point_resampling', False):
        train_transforms_list.append(PointResamplingTransform(resample_ratio=config.training.augment_resampling_ratio))

    train_transform = PointCloudAugmentation(train_transforms_list) if train_transforms_list else None
    if train_transform:
        logger.info(f"Using {len(train_transforms_list)} training augmentations.")

    train_dataset = PointCloudDataset(root=config.data.splits.train_dir, transform=train_transform)
    val_dataset = PointCloudDataset(root=config.data.splits.val_dir)
    test_dataset = PointCloudDataset(root=config.data.splits.test_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")

    # --- 3. Build BI-Net model & Handle Checkpoint ---
    logger.info("[INFO] Building model...")
    binet = BiNet(
        latent_dim=config.model.latent_dim,
        features_g=config.model.features_g,
        degrees_g=config.model.degrees,
        support=config.model.support,
        dropout_rate=getattr(config.training, 'dropout_rate', 0.1),
        use_spectral_norm=getattr(config.training, 'use_spectral_norm', False)
    ).to(device)

    save_ckpt_path = None
    if ckpt_name:
        initial_ckpt_path = os.path.join(config.model.save_dir, ckpt_name)
        if os.path.exists(initial_ckpt_path):
            logger.info(f"Attempting to load BI-Net checkpoint from {initial_ckpt_path}")
            try:
                binet.load_state_dict(torch.load(initial_ckpt_path, map_location=device))
                logger.info("Successfully loaded checkpoint.")
                save_ckpt_path = initial_ckpt_path
            except Exception as e:
                logger.error(f"Failed to load checkpoint {initial_ckpt_path}: {e}")
                if not do_train:
                    logger.error("Cannot proceed with evaluation/generation without a valid model. Exiting.")
                    return
                else:
                    logger.warning("Could not load checkpoint. Training from scratch.")
        else:
            logger.warning(f"Provided checkpoint '{initial_ckpt_path}' not found.")
            if not do_train:
                logger.error("Cannot proceed with evaluation/generation as specified checkpoint was not found. Exiting.")
                return
            save_ckpt_path = initial_ckpt_path
    
    # --- 4. Train Model ---
    if do_train:
        if not save_ckpt_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_ckpt_name = f"bi_net_epoch{config.training.epochs}_{timestamp}.pth"
            save_ckpt_path = os.path.join(config.model.save_dir, save_ckpt_name)
        
        logger.info(f"Starting training. Model will be saved to: {save_ckpt_path}")
        
        binet = train_binet(
            model=binet,
            train_loader=train_loader,
            config=config,
            val_loader=val_loader,
            logger=logger,
        )
        
        logger.info(f"Training finished. Saving final checkpoint to {save_ckpt_path}")
        os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
        torch.save(binet.state_dict(), save_ckpt_path)

    # --- 5. Evaluate Model ---
    if do_eval:
        logger.info("[INFO] Evaluating final model on the test set.")
        avg_emd, avg_chamfer = evaluate_on_loader_emd_chamfer(binet, test_loader, device)
        logger.info(f"Test Set Evaluation Result -> EMD: {avg_emd:.4f}, Chamfer: {avg_chamfer:.4f}")

    # --- 6. Generate New Shapes ---
    if do_generate:
        logger.info("[INFO] Generating random shapes from the trained model.")
        binet.eval()
        sample_count = config.generation.sample_count
        z = torch.randn(sample_count, config.model.latent_dim, device=device)
        with torch.no_grad():
            fake_points = binet.generate(z)
        logger.info(f"Generated {fake_points.shape[0]} shapes of size {fake_points.shape[1]}x{fake_points.shape[2]}")

        out_gen_path = os.path.join(config.model.save_dir, "generated_shapes.npz")
        np.savez(out_gen_path, points=fake_points.cpu().numpy())
        logger.info(f"Saved generated shapes to {out_gen_path}")