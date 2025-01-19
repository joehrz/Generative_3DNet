"""
Unified entry point for BI-Net pipeline:
  - Preprocess raw data (if requested)
  - (Optionally) split processed data into train/val/test
  - (Optionally) train the BI-Net
  - (Optionally) evaluate reconstruction or generate shapes

Usage Example:
  python src/main.py --preprocess \
      --input_dir data/raw \
      --output_dir data/processed \
      --voxel_size 0.02 --num_points 2048 --use_fps
  python src/main.py --split
  python src/main.py --train --data_dir data/splits
  ...
"""

import os
import sys
import argparse
import shutil
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data_processing.data_processing import preprocess_point_clouds
from src.datasets.pc_dataset import PointCloudDataset
from src.utils.logger import setup_logger
from src.configs.config import Config
from src.data_processing.dataset_splitting import split_dataset
from src.models.bi_net import BiNet
from src.utils.train_utils import train_binet
from src.utils.emd import emd_loss
from src.utils.pc_utils import compute_chamfer_distance 

def parse_args():
    parser = argparse.ArgumentParser(description="BI-Net 3D Point Cloud Pipeline")

    # Preprocessing flags
    parser.add_argument("--preprocess", action="store_true",
                        help="Run data preprocessing (downsample, unify points, normalize).")
    parser.add_argument("--input_dir", type=str, default="data/raw",
                        help="Input directory of raw point clouds (for preprocessing).")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output directory for processed point clouds.")
    parser.add_argument("--voxel_size", type=float, default=0.02,
                        help="Voxel size for downsampling.")
    parser.add_argument("--num_points", type=int, default=2048,
                        help="Number of points after unify step.")
    parser.add_argument("--use_fps", action="store_true",
                        help="Use farthest point sampling to preserve structure.")

    # Splitting flags
    parser.add_argument("--split", action="store_true",
                        help="Split the processed data into train/val/test in data/splits.")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Ratio of data to go into train split.")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Ratio of data to go into val split.")
    # test ratio will be 1 - (train_ratio + val_ratio)

    # Flags for training or evaluation
    parser.add_argument("--train", action="store_true", help="Train the BI-Net.")
    parser.add_argument("--eval", action="store_true", help="Evaluate AE reconstruction.")
    parser.add_argument("--generate", action="store_true", help="Generate shapes from random noise.")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device.")

    # Model hyperparams
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--checkpoint", type=str, default="bi_net_checkpoint.pth",
                        help="Path to save/load model checkpoint.")
    parser.add_argument("--data_dir", type=str, default="data/splits",
                        help="Where to load final splits for train/test.")
    return parser.parse_args()




def main():
    args = parse_args()
    # Load configuration
    config = Config(args.config)
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "pipeline.log")
    logger = setup_logger("pipeline_logger", log_file)

    #######################################
    # 1) Preprocess (Optional)
    #######################################
    if args.preprocess:
        logger.info(f"Preprocessing from {config.data.raw_dir} => {config.data.processed_dir}, voxel={config.preprocessing.voxel_size}, "
              f"points={config.preprocessing.num_points}, use_fps={config.preprocessing.use_fps}")
        preprocess_point_clouds(
            input_dir=config.data.raw_dir,   # e.g. "data/raw"
            output_dir=config.data.processed_dir,  # e.g. "data/processed"         
            voxel_size=config.preprocessing.voxel_size,
            num_points=config.preprocessing.num_points,
            file_ext=".npz",  # or ".npz" depending on your pipeline
            use_fps=config.preprocessing.use_fps
        )
        logger.info("Preprocessing completed.")

    #######################################
    # 2) Split data into train/val/test (Optional)
    #######################################

    # 4) Dataset Splitting => data/processed -> data/processed/splits
    if args.split:
        logger.info("Starting dataset splitting...")
        # If you want to split the final preprocessed data:
        input_dir = config.data.processed_dir
        output_dir = os.path.join(config.data.processed_dir, "splits")
        split_ratios = tuple(config.data.split_ratios)
        logger.info(f"Splitting data from {input_dir} => {output_dir}")
        split_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            split_ratios=split_ratios
        )
        logger.info("Dataset splitting completed.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    #######################################
    # 3) (Optional) Train or Evaluate
    #######################################
    # If you're using splitted data, assume 'args.data_dir' => "data/splits"
    # with subfolders train/, val/, test/
    if args.train or args.eval or args.generate:
        # Example usage: data_dir => "data/splits"
        # We'll load from train => data_dir/train
        from torch.utils.data import DataLoader

        print(f"Loading train dataset from {args.data_dir}/train ...")
        train_dataset = PointCloudDataset(root=args.data_dir, split='train')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Optionally test dataset
        test_dataset = PointCloudDataset(root=args.data_dir, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Build BI-Net
        features_g = [args.latent_dim, 128, 64, 3]
        degrees = [4, 4, 4]
        enc_disc_feat = [3, 64, 128, 256, 512]

        binet = BiNet(
            batch_size=args.batch_size,
            features_g=features_g,
            degrees=degrees,
            enc_disc_feat=enc_disc_feat,
            latent_dim=args.latent_dim,
            support=10
        ).to(device)

        # If checkpoint exists, load it (if not training from scratch)
        if os.path.exists(args.checkpoint):
            binet.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Loaded BI-Net checkpoint from {args.checkpoint}")

        # Train
        if args.train:
            print(f"Training BI-Net for {args.epochs} epochs ...")
            train_binet(
                binet,
                data_loader=train_loader,
                device=device,
                epochs=args.epochs,
                latent_dim=args.latent_dim,
                lambda_gp=10.0,
                lambda_nnme=0.05
            )
            torch.save(binet.state_dict(), args.checkpoint)
            print(f"Training done. Checkpoint saved to {args.checkpoint}")

        # Evaluate
        if args.eval:
            print("Evaluating AE reconstruction on test set ...")
            binet.eval()
            total_emd, count = 0.0, 0
            with torch.no_grad():
                for real_points in test_loader:
                    real_points = real_points.to(device)
                    B = real_points.size(0)

                    latent = binet.encode(real_points)
                    rec_points = binet.decode(latent)
                    emd_val = emd_loss(rec_points, real_points).item()
                    total_emd += emd_val * B
                    count += B

            avg_emd = total_emd / max(count, 1)
            print(f"Avg EMD (placeholder) on test: {avg_emd:.4f}")

        # Generate shapes
        if args.generate:
            print("Generating random shapes ...")
            binet.eval()
            B = 2  # example: generate 2 shapes
            z = torch.randn(B, args.latent_dim, device=device)
            with torch.no_grad():
                fake_points = binet.generate(z)
            print(f"Generated shape: {fake_points.shape}")
            # Save or visualize if needed.

    print("Pipeline finished.")

if __name__ == "__main__":
    main()


