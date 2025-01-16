# File: scripts/evaluate.py

import argparse
import torch
from torch.utils.data import DataLoader

from src.models.bi_net import BiNet
from src.datasets.pc_dataset import WheatPointCloudDataset
from src.utils.emd import emd_loss  # or real EMD/Chamfer if you have it
from src.utils.pc_utils import compute_chamfer_distance  # Example placeholder

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the BI-Net on a test set.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--latent_dim', type=int, default=96)
    parser.add_argument('--model_checkpoint', type=str, default='bi_net_checkpoint.pth',
                        help='Path to the trained model weights')
    parser.add_argument('--data_root', type=str, default='data/processed')
    parser.add_argument('--split', type=str, default='test', help='Which split to evaluate on')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Example: define your BI-Net with the same hyperparams used in training
    batch_size = args.batch_size
    features_g = [96, 128, 64, 3]
    degrees = [4, 4, 4]
    enc_disc_feat = [3, 64, 128, 256, 512]
    latent_dim = args.latent_dim
    support = 10

    binet = BiNet(
        batch_size=batch_size,
        features_g=features_g,
        degrees=degrees,
        enc_disc_feat=enc_disc_feat,
        latent_dim=latent_dim,
        support=support
    )
    binet.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    binet.to(device)
    binet.eval()

    # Load test dataset
    test_dataset = WheatPointCloudDataset(root=args.data_root, split=args.split, transform=None, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    total_emd = 0.0
    total_chamfer = 0.0
    count = 0

    with torch.no_grad():
        for real_points in test_loader:
            real_points = real_points.to(device)
            B = real_points.size(0)

            # AE direction: reconstruct
            latent_code = binet.encode(real_points)
            rec_points = binet.decode(latent_code)

            # EMD or Chamfer
            emd_val = emd_loss(rec_points, real_points).item()  # Placeholder
            chamfer_val = compute_chamfer_distance(rec_points, real_points).item()  # If you have an implementation

            total_emd += emd_val * B
            total_chamfer += chamfer_val * B
            count += B

    avg_emd = total_emd / count
    avg_chamfer = total_chamfer / count
    print(f"Evaluation on {args.split} split - EMD: {avg_emd:.6f}, Chamfer: {avg_chamfer:.6f}")

if __name__ == "__main__":
    main()

