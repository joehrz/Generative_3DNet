# File: main.py

"""
Unified entry point for the BI-Net pipeline.

This script acts as a controller for the entire workflow, from data preparation
to model training and inference. It uses a YAML configuration file for managing
all parameters, making experiments reproducible and easy to configure.

Core functionalities are triggered by command-line flags:
  --preprocess: Kicks off the data preparation stage.
  --split:      Splits the prepared data into train/validation/test sets.
  --train:      Starts the model training process.
  --eval:       Evaluates a trained model on the test set.
  --generate:   Generates new point cloud shapes from a trained model.

Usage Examples:
-----------------------------------------------------------------------------
# 1. Preprocess the raw data as defined in the config file
(Edit default_config.yaml to set your raw_dir and processed_dir)
$ python main.py --config src/configs/default_config.yaml --preprocess

# 2. Split the processed data into train/val/test sets
$ python main.py --config src/configs/default_config.yaml --split

# 3. Train the model using the data splits and training params from the config
$ python main.py --config src/configs/default_config.yaml --train

# 4. Train a model and save it with a specific checkpoint name
$ python main.py --config src/configs/default_config.yaml --train --checkpoint my_sorghum_model.pth

# 5. Evaluate a trained model
$ python main.py --config src/configs/default_config.yaml --eval --checkpoint my_sorghum_model.pth

# 6. Run all steps in sequence
$ python main.py --config src/configs/default_config.yaml --preprocess --split --train --eval
-----------------------------------------------------------------------------
"""

import os
import sys
import argparse

# This robustly sets the project root, assuming main.py is in a 'scripts'
# or 'src' subdirectory, or even at the top level.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset.data_preprocessing import preprocess_point_clouds
from src.dataset.dataset_splitting import split_dataset
from src.utils.logger import setup_logger
from src.configs.config import Config
from src.training.training import run_training_pipeline

def parse_args():
    """
    Parses command-line arguments.

    The arguments are designed to be high-level "actions" that control which
    parts of the pipeline to run. All detailed parameters are managed by the
    config file for better reproducibility.
    """
    parser = argparse.ArgumentParser(
        description="BI-Net 3D Point Cloud Pipeline",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/default_config.yaml",
        help="Path to the YAML configuration file."
    )
    
    # --- Action Flags ---
    parser.add_argument("--preprocess", action="store_true", help="Run data preprocessing.")
    parser.add_argument("--split", action="store_true", help="Split the processed data.")
    parser.add_argument("--train", action="store_true", help="Train the BI-Net model.")
    parser.add_argument("--eval", action="store_true", help="Evaluate a trained model.")
    parser.add_argument("--generate", action="store_true", help="Generate shapes from random noise.")
    
    # --- Overrides & General Options ---
    parser.add_argument(
        "--device",
        type=str,
        default=None, # Default to None, will use config value if not provided
        help="Override compute device (e.g., 'cuda', 'cpu'). Defaults to value in config."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specify a checkpoint file name (relative to 'save_dir' in config) to load or save."
    )
    
    return parser.parse_args()

def main():
    """
    Main execution function of the pipeline.
    """
    args = parse_args()
    
    # --- 1. Load Configuration ---
    try:
        config = Config(args.config)
    except FileNotFoundError:
        print(f"[ERROR] Configuration file not found at: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load or parse configuration file: {e}")
        sys.exit(1)

    # --- 2. Setup Logging ---
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "pipeline.log")
    logger = setup_logger("pipeline_logger", log_file)
    logger.info("=================================================")
    logger.info("=== Starting BI-Net Pipeline Execution        ===")
    logger.info("=================================================")
    logger.info(f"Loaded configuration from: {args.config}")

    # --- 3. Execute Pipeline Stages based on Flags ---

    # Stage 3a: Preprocessing
    if args.preprocess:
        logger.info("--- Stage: Preprocessing ---")
        logger.info(f"Input dir: {config.data.raw_dir}, Output dir: {config.data.processed_dir}")
        try:
            preprocess_point_clouds(
                input_dir=config.data.raw_dir,
                output_dir=config.data.processed_dir,
                voxel_size=config.preprocessing.voxel_size,
                num_points=config.preprocessing.num_points,
                use_fps=config.preprocessing.use_fps,
                skip_downsample=config.preprocessing.get('skip_downsample', False)
            )
            logger.info("Preprocessing completed successfully.")
        except Exception as e:
            logger.error(f"An error occurred during preprocessing: {e}", exc_info=True)
            sys.exit(1)

    # Stage 3b: Dataset Splitting
    if args.split:
        logger.info("--- Stage: Dataset Splitting ---")
        logger.info(f"Splitting data from {config.data.processed_dir} into {config.data.splits_dir}")
        try:
            split_dataset(
                input_dir=config.data.processed_dir,
                output_dir=config.data.splits_dir,
                split_ratios=tuple(config.data.split_ratios)
            )
            logger.info("Dataset splitting completed successfully.")
        except Exception as e:
            logger.error(f"An error occurred during dataset splitting: {e}", exc_info=True)
            sys.exit(1)

    # Stage 3c: Training, Evaluation, or Generation
    if args.train or args.eval or args.generate:
        logger.info("--- Stage: Model Training / Inference ---")
        
        # Override config with command-line args if provided
        if args.device:
            config.training.device = args.device
            logger.info(f"Overriding device with command-line argument: {args.device}")

        try:
            run_training_pipeline(
                config=config,
                logger=logger,
                do_train=args.train,
                do_eval=args.eval,
                do_generate=args.generate,
                ckpt_name=args.checkpoint
            )
            logger.info("Training/Inference pipeline finished successfully.")
        except Exception as e:
            logger.error(f"An error occurred during the training/inference pipeline: {e}", exc_info=True)
            sys.exit(1)

    logger.info("=================================================")
    logger.info("=== BI-Net Pipeline Finished                  ===")
    logger.info("=================================================")
    print("Script finished. Check 'logs/pipeline.log' for details.")

if __name__ == "__main__":
    main()
