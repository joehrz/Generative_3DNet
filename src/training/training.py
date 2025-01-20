# File: src/training/training.py

import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.dataset.pc_dataset import PointCloudDataset
from src.models.bi_net import BiNet
from src.utils.train_utils import train_binet
from src.utils.emd import emd_loss
from src.utils.pc_utils import compute_chamfer_distance
from src.configs.config import Config

def run_pipeline(config, logger, do_train=False, do_eval=False, do_generate=False):
    """
    High-level routine that:
      1) Builds dataset loaders from config paths (train/val/test).
      2) Builds & loads BI-Net model from config.
      3) Optionally trains the model (if do_train=True).
      4) Optionally evaluates (if do_eval=True).
      5) Optionally generates shapes from random noise (if do_generate=True).

    Args:
        config (Config): Loaded YAML config object
        logger: Logger object for logging info
        do_train (bool): Whether to run training
        do_eval (bool): Whether to run evaluation on val or test set
        do_generate (bool): Whether to generate shapes from random latent vectors
    """
    # -----------------------------
    # 1. Device configuration
    # -----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[INFO] Using device: {device}")


    # -----------------------------
    # 2. Datasets & Dataloaders
    # -----------------------------
    train_dir = config.data.splits.train_dir
    val_dir   = config.data.splits.val_dir
    test_dir  = config.data.splits.test_dir

    # Gather file paths for each split
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
    val_files   = sorted(glob.glob(os.path.join(val_dir, "*.npz")))
    test_files  = sorted(glob.glob(os.path.join(test_dir, "*.npz")))

    # Create datasets. 
    # If your PointCloudDataset init expects a list of files, pass them in:
    train_dataset = PointCloudDataset(
        files=train_files,                        # or however your pc_dataset is structured
        num_points=config.model.final_num_points  # e.g. 2048 or 64 
    )
    val_dataset = PointCloudDataset(
        files=val_files,
        num_points=config.model.final_num_points
    )

    # Alternatively, if your dataset constructor uses (root=..., split=...),
    # you can do:
    #   train_dataset = PointCloudDataset(root=config.data.splits_dir, split="train")
    #   val_dataset   = PointCloudDataset(root=config.data.splits_dir, split="val")
    #   etc.

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,     # adjust as needed
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # If you want a test loader, do similarly:
    test_dataset = PointCloudDataset(
        files=test_files,
        num_points=config.model.final_num_points
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} val samples.")

    # -----------------------------
    # 3. Build BI-Net model
    # -----------------------------

    latent_dim = config.model.latent_dim
    batch_size = config.training.batch_size

    features_g = config.model.features_g   # e.g. [96, 128, 64, 3]
    degrees    = config.model.degrees      # e.g. [4,4,4]
    enc_disc_feat = config.model.enc_disc_feat  # e.g. [3,64,128,256,512]
    support = config.model.support


    binet = BiNet(
        batch_size=batch_size,
        features_g=features_g,
        degrees=degrees,
        enc_disc_feat=enc_disc_feat,
        latent_dim=latent_dim,
        support=support
    ).to(device)


    # Load checkpoint if needed
    ckpt_path = os.path.join(config.model.save_dir, config.model.checkpoint_name)
    if os.path.exists(ckpt_path):
        logger.info(f"Loading BI-Net checkpoint from {ckpt_path}")
        binet.load_state_dict(torch.load(ckpt_path, map_location=device))



    # -----------------------------
    # 4. Train if do_train=True
    # -----------------------------
    if do_train:
        logger.info(f"[INFO] Starting training for {config.training.epochs} epochs...")
        binet = train_binet(
            binet,
            data_loader=train_loader,
            device=device,
            epochs=config.training.epochs,
            latent_dim=latent_dim,
            lambda_gp=config.model.lambda_gp,
            lambda_nnme=config.model.lambda_nnme,
            logger=logger
        )
        # Save checkpoint
        logger.info("[INFO] Training done. Saving checkpoint.")
        torch.save(binet.state_dict(), ckpt_path)

    # -----------------------------
    # 5. Evaluate if do_eval=True
    # -----------------------------
    if do_eval:
        logger.info("[INFO] Evaluating reconstruction on test (or val) set...")
        binet.eval()
        total_emd = 0.0
        total_chamfer = 0.0
        count = 0

        with torch.no_grad():
            for real_points in test_loader:  # or val_loader if you prefer
                real_points = real_points.to(device)
                B = real_points.size(0)

                # AE direction
                latent_code = binet.encode(real_points)
                rec_points  = binet.decode(latent_code)

                # EMD or Chamfer
                emd_val = emd_loss(rec_points, real_points).item()
                chamfer_val = compute_chamfer_distance(rec_points, real_points).item()

                total_emd      += emd_val * B
                total_chamfer  += chamfer_val * B
                count          += B

        avg_emd      = total_emd / max(count, 1)
        avg_chamfer  = total_chamfer / max(count, 1)
        logger.info(f"[RESULT] EMD: {avg_emd:.4f}, Chamfer: {avg_chamfer:.4f}")

    # -----------------------------
    # 6. Generate if do_generate=True
    # -----------------------------
    if do_generate:
        logger.info("[INFO] Generating random shapes from latent noise.")
        binet.eval()
        sample_count = 4  # e.g. generate 4 shapes
        z = torch.randn(sample_count, latent_dim, device=device)
        with torch.no_grad():
            fake_points = binet.generate(z)
        logger.info(f"[INFO] Generated shape => {fake_points.shape}  (B, N, 3)")

        # You might want to save them to .npz, or do something else:
        out_gen_path = os.path.join(config.model.save_dir, "generated_points.npz")
        np.savez(out_gen_path, points=fake_points.cpu().numpy())
        logger.info(f"[INFO] Saved generated points to {out_gen_path}")