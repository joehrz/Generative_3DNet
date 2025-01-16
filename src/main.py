"""
Unified entry point for BI-Net pipeline:
  - Load dataset
  - Train the BI-Net
  - (Optional) Evaluate reconstruction
  - (Optional) Generate new shapes
Usage Example:
  python src/main.py --data_dir data/processed --train --eval --generate
"""

import os
import sys
import argparse
import torch

# so we can import from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.datasets.pc_dataset import WheatPointCloudDataset
from src.models.bi_net import BiNet
from src.utils.train_utils import train_binet
from src.utils.emd import emd_loss
from src.utils.pc_utils import compute_chamfer_distance  # if you have it

from src.data_processing.data_processing import preprocess_point_clouds

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

    # Flags for training or evaluation
    parser.add_argument("--train", action="store_true", help="Train the BI-Net.")
    parser.add_argument("--eval", action="store_true", help="Evaluate AE reconstruction.")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device.")

    parser.add_argument("--generate", action="store_true", help="Flag to generate new shapes from noise.")
    parser.add_argument("--checkpoint", type=str, default="bi_net_checkpoint.pth",
                        help="Path to save/load model checkpoint.")

    return parser.parse_args()

def main():
    args = parse_args()
    # 1) Preprocessing
    if args.preprocess:
        print(f"Preprocessing from {args.input_dir} => {args.output_dir}, voxel={args.voxel_size}, "
              f"points={args.num_points}, use_fps={args.use_fps}")
        preprocess_point_clouds(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            voxel_size=args.voxel_size,
            num_points=args.num_points,
            file_ext=".npy",     # or ".npy"
            use_fps=args.use_fps
        )
        print("Preprocessing completed.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) Load Dataset
    print(f"Loading train dataset from {args.data_dir}/train ...")
    train_dataset = WheatPointCloudDataset(root=args.data_dir, split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Optionally load test dataset if you want to evaluate or test
    test_dataset = WheatPointCloudDataset(root=args.data_dir, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2) Build BI-Net
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

    # 3) Train if requested
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
    else:
        # If not training, but checkpoint exists, load it
        if os.path.exists(args.checkpoint):
            binet.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Loaded BI-Net checkpoint from {args.checkpoint}")

    # 4) Evaluate AE reconstruction if requested
    if args.eval:
        print("Evaluating AE reconstruction on test set ...")
        binet.eval()
        total_emd, total_chamfer, count = 0.0, 0.0, 0
        with torch.no_grad():
            for real_points in test_loader:
                real_points = real_points.to(device)
                B = real_points.size(0)

                latent = binet.encode(real_points)
                rec_points = binet.decode(latent)
                emd_val = emd_loss(rec_points, real_points).item()
                total_emd += emd_val * B

                # If you have a chamfer function
                # chamfer_val = compute_chamfer_distance(rec_points, real_points).item()
                # total_chamfer += chamfer_val * B

                count += B

        avg_emd = total_emd / max(count, 1)
        # avg_chamfer = total_chamfer / max(count, 1) if needed
        print(f"Avg EMD (placeholder) on test: {avg_emd:.4f}")

    # 5) Generate shapes if requested
    if args.generate:
        print("Generating random shapes ...")
        binet.eval()
        B = 2  # example: generate 2 shapes
        z = torch.randn(B, args.latent_dim, device=device)
        with torch.no_grad():
            fake_points = binet.generate(z)
        print(f"Generated shape: {fake_points.shape}")

        # e.g., save or visualize them
        # In a real scenario, you might do something like:
        # np.save("generated_points.npy", fake_points.cpu().numpy())

    print("Pipeline finished.")

if __name__ == "__main__":
    main()

