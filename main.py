"""
Unified entry point for BI-Net pipeline:
  - Preprocess raw data (if requested)
  - (Optionally) split processed data into train/val/test
  - (Optionally) train the BI-Net
  - (Optionally) evaluate reconstruction or generate shapes

Usage Example:
  python main.py --preprocess \
      --input_dir data/raw \
      --output_dir data/processed \
      --voxel_size 0.02 --num_points 2048 --use_fps
  python main.py --split
  python main.py --train --data_dir data/splits
  ...
"""

import os
import sys
import argparse


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.dataset.data_preprocessing import preprocess_point_clouds
from src.dataset.dataset_splitting import split_dataset
from src.utils.logger import setup_logger
from src.configs.config import Config
from src.training.training import run_training_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="BI-Net 3D Point Cloud Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="Path to the configuration file.")
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
    parser.add_argument("--skip_downsample", action="store_true",
                        help="Skip voxel downsampling entirely if set.")

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
    parser.add_argument("--eval", action="store_true", help="Evaluate model.")
    parser.add_argument("--generate", action="store_true", help="Generate shapes from random noise.")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device.")

    # Model hyperparams
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=128)
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
    # 1) Preprocess
    #######################################
    if args.preprocess:
        logger.info(f"Preprocessing from {config.data.raw_dir} => {config.data.processed_dir}, voxel={config.preprocessing.voxel_size}, "
              f"points={config.preprocessing.num_points}, use_fps={config.preprocessing.use_fps}")
        preprocess_point_clouds(
            input_dir=config.data.raw_dir,   # e.g. "data/raw"
            output_dir=config.data.processed_dir,  # e.g. "data/processed"         
            voxel_size=config.preprocessing.voxel_size,
            num_points=config.preprocessing.num_points,
            use_fps=config.preprocessing.use_fps,
            skip_downsample=args.skip_downsample
            )
        logger.info("Preprocessing completed.")

    #######################################
    # 2) Split data into train/val/test
    #######################################

    # 4) Dataset Splitting => data/processed -> data/processed/splits
    if args.split:
        logger.info("Starting dataset splitting...")
        # If you want to split the final preprocessed data:
        input_dir = config.data.processed_dir
        output_dir = config.data.splits_dir
        split_ratios = tuple(config.data.split_ratios)
        logger.info(f"Splitting data from {input_dir} => {output_dir}")
        split_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            split_ratios=split_ratios
        )
        logger.info("Dataset splitting completed.")

    #######################################
    # 3) Train/Eval/Generate
    #######################################
    # If any of train, eval, generate is set, we call run_training_pipeline
    # We'll pass do_train, do_eval, do_generate based on the flags.
    if args.train or args.eval or args.generate:
        logger.info("[MAIN] Running training pipeline with possible eval/generate.")
        run_training_pipeline(
            config=config,
            logger=logger,
            do_train=args.train,
            do_eval=args.eval,
            do_generate=args.generate,
            ckpt_name=args.checkpoint
        )
        logger.info("[MAIN] Training/Eval/Generation pipeline finished.")

    logger.info("=== BI-Net Pipeline Finished ===")
    print("Pipeline finished.")


if __name__ == "__main__":
    main()