# File: src/training/training.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.dataset.pc_dataset import PointCloudDataset
from src.models.bi_net import BiNet
from src.utils.train_utils import train_binet
from src.utils.losses import evaluate_on_loader_emd_chamfer


#PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#sys.path.append(PROJECT_ROOT)

def run_training_pipeline(config, logger, do_train=False, do_eval=False, do_generate=False):
    """
    High-level routine that:
      1) Builds dataset loaders from config paths (train/val/test).
      2) Builds & loads BI-Net model from config.
      3) Optionally trains (do_train=True) with in-line validation
      4) Optionally does final evaluation (do_eval=True)
      5) Optionally generates shapes (do_generate=True)
    """
    # -----------------------------
    # 1. Device configuration
    # -----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[INFO] Using device: {device}")


    # -----------------------------
    # 2. Datasets & Dataloaders
    # -----------------------------


    train_dataset = PointCloudDataset(
        root=config.data.splits.train_dir,
        )
    val_dataset = PointCloudDataset(
        root=config.data.splits.val_dir,
        )

    test_dataset = PointCloudDataset(
        root=config.data.splits.test_dir
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.training.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    logger.info(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")

    # -----------------------------
    # 3. Build BI-Net model
    # -----------------------------

    latent_dim    = config.model.latent_dim
    batch_size    = config.training.batch_size
    features_g    = config.model.features_g
    degrees       = config.model.degrees
    ae_enc_feat = config.model.ae_enc_feat
    disc_hidden   = config.model.disc_hidden
    support       = config.model.support

    binet = BiNet(
        batch_size=batch_size,
        ae_enc_feat=ae_enc_feat,
        latent_dim=latent_dim,
        disc_hidden=disc_hidden,
        features_g=features_g,
        degrees=degrees,
        support=support
    ).to(device)

    # Load checkpoint if needed
    ckpt_path = os.path.join(config.model.save_dir, config.model.checkpoint_name)
    print(ckpt_path)
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
            val_loader=val_loader,   # pass val_loader here
            device=device,
            epochs=config.training.epochs,
            latent_dim=latent_dim,
            lambda_gp=config.model.lambda_gp,
            lambda_nnme=config.model.lambda_nnme,
            logger=logger,
            # you can do val_interval=1 so we validate every epoch
            # or val_interval=2, etc.
            val_interval=1
        )
        logger.info("[INFO] Training done. Saving checkpoint.")
        torch.save(binet.state_dict(), ckpt_path)

    # -----------------------------
    # 5. Evaluate if do_eval=True
    # -----------------------------
    if do_eval:
        logger.info("[INFO] Evaluating on the test set (final check).")
        binet.eval()
        avg_emd, avg_chamfer = evaluate_on_loader_emd_chamfer(binet, test_loader, device)
        logger.info(f"[RESULT] Test EMD: {avg_emd:.4f}, Chamfer: {avg_chamfer:.4f}")


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