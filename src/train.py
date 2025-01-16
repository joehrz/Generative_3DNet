import torch
from torch.utils.data import DataLoader

# Project imports
from src.models.bi_net import BiNet
from src.utils.train_utils import train_binet
from src.datasets.pc_dataset import PointCloudDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    # Instantiate dataset
    # Adjust root path and split as needed
    train_dataset = PointCloudDataset(root='data/processed', split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Model hyperparameters
    batch_size = 8
    features_g = [96, 128, 64, 3]
    degrees = [4, 4, 4]
    enc_disc_feat = [3, 64, 128, 256, 512]  # for EnDi
    latent_dim = 96
    support = 10

    # Create BI-Net
    binet = BiNet(
        batch_size=batch_size,
        features_g=features_g,
        degrees=degrees,
        enc_disc_feat=enc_disc_feat,
        latent_dim=latent_dim,
        support=support
    )

    # Train
    trained_model = train_binet(
        binet,
        data_loader=train_loader,
        device=device,
        epochs=10,
        latent_dim=latent_dim,
        lambda_gp=10.0,
        lambda_nnme=0.05,
        lr_enc=1e-4,
        lr_dec=1e-4,
        lr_disc=5e-5,
        d_iters=1,
        g_iters=1
    )

    # Save checkpoint
    torch.save(trained_model.state_dict(), "bi_net_checkpoint.pth")
    print("Training complete. Model saved as bi_net_checkpoint.pth")

if __name__ == "__main__":
    main()