# File: src/utils/cross_validation.py

"""
K-Fold Cross-Validation utilities for small dataset training.
"""

import os
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import copy

from src.models.bi_net import BiNet
from src.utils.train_utils import train_binet
from src.utils.losses import evaluate_on_loader_emd_chamfer


def k_fold_cross_validation(dataset, config, logger, k_folds=5, random_state=42):
    """
    Perform k-fold cross-validation for small dataset training.
    
    Args:
        dataset: The full dataset to split
        config: Training configuration
        logger: Logger instance
        k_folds: Number of folds for cross-validation
        random_state: Random seed for reproducibility
        
    Returns:
        Dict with cross-validation results
    """
    logger.info(f"Starting {k_folds}-fold cross-validation...")
    
    # Initialize k-fold splitter
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    # Store results for each fold
    fold_results = []
    best_models = []
    
    # Get dataset indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    
    for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        logger.info(f"--- Fold {fold + 1}/{k_folds} ---")
        logger.info(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        
        # Create train and validation subsets
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, 
            batch_size=config.training.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, 
            batch_size=config.training.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        # Initialize model for this fold
        binet = BiNet(
            latent_dim=config.model.latent_dim,
            features_g=config.model.features_g,
            degrees_g=config.model.degrees,
            support=config.model.support,
            dropout_rate=getattr(config.training, 'dropout_rate', 0.1),
            use_spectral_norm=getattr(config.training, 'use_spectral_norm', False)
        ).to(device)
        
        # Train model for this fold
        fold_config = copy.deepcopy(config)
        fold_config.training.epochs = max(config.training.epochs // 2, 20)  # Shorter training per fold
        
        logger.info(f"Training fold {fold + 1} for {fold_config.training.epochs} epochs...")
        
        try:
            # Train the model
            train_binet(
                binet=binet,
                train_loader=train_loader,
                val_loader=val_loader,
                config=fold_config,
                logger=logger,
                save_ckpt_path=None  # Don't save individual fold checkpoints
            )
            
            # Evaluate on validation set
            val_emd, val_chamfer = evaluate_on_loader_emd_chamfer(
                binet, val_loader, device, 
                config.training.emd_eps, config.training.emd_iters
            )
            
            fold_result = {
                'fold': fold + 1,
                'val_emd': val_emd,
                'val_chamfer': val_chamfer,
                'train_size': len(train_indices),
                'val_size': len(val_indices)
            }
            fold_results.append(fold_result)
            best_models.append(copy.deepcopy(binet.state_dict()))
            
            logger.info(f"Fold {fold + 1} Results - EMD: {val_emd:.6f}, Chamfer: {val_chamfer:.6f}")
            
        except Exception as e:
            logger.error(f"Error in fold {fold + 1}: {e}")
            fold_result = {
                'fold': fold + 1,
                'val_emd': float('inf'),
                'val_chamfer': float('inf'),
                'train_size': len(train_indices),
                'val_size': len(val_indices),
                'error': str(e)
            }
            fold_results.append(fold_result)
            best_models.append(None)
    
    # Calculate cross-validation statistics
    valid_results = [r for r in fold_results if 'error' not in r]
    
    if valid_results:
        emd_scores = [r['val_emd'] for r in valid_results]
        chamfer_scores = [r['val_chamfer'] for r in valid_results]
        
        cv_stats = {
            'mean_emd': np.mean(emd_scores),
            'std_emd': np.std(emd_scores),
            'mean_chamfer': np.mean(chamfer_scores),
            'std_chamfer': np.std(chamfer_scores),
            'best_fold': np.argmin(emd_scores) + 1,
            'best_emd': np.min(emd_scores),
            'valid_folds': len(valid_results),
            'total_folds': k_folds
        }
        
        logger.info("=== Cross-Validation Results ===")
        logger.info(f"Mean EMD: {cv_stats['mean_emd']:.6f} ± {cv_stats['std_emd']:.6f}")
        logger.info(f"Mean Chamfer: {cv_stats['mean_chamfer']:.6f} ± {cv_stats['std_chamfer']:.6f}")
        logger.info(f"Best fold: {cv_stats['best_fold']} (EMD: {cv_stats['best_emd']:.6f})")
        logger.info(f"Valid folds: {cv_stats['valid_folds']}/{cv_stats['total_folds']}")
    else:
        logger.error("No valid folds completed successfully!")
        cv_stats = None
    
    return {
        'fold_results': fold_results,
        'cv_stats': cv_stats,
        'best_models': best_models
    }


def ensemble_predict(models, data_loader, device, config):
    """
    Make ensemble predictions using multiple models from cross-validation.
    
    Args:
        models: List of model state dicts from different folds
        data_loader: DataLoader for the test data
        device: Device to run inference on
        config: Configuration object
        
    Returns:
        Ensemble predictions
    """
    valid_models = [m for m in models if m is not None]
    if not valid_models:
        raise ValueError("No valid models available for ensemble prediction")
    
    ensemble_results = []
    
    for batch_idx, batch in enumerate(data_loader):
        if isinstance(batch, (list, tuple)):
            real_points = batch[0].to(device)
        else:
            real_points = batch.to(device)
        
        # Get predictions from each model
        model_predictions = []
        
        for model_state in valid_models:
            # Load model
            binet = BiNet(
                latent_dim=config.model.latent_dim,
                features_g=config.model.features_g,
                degrees_g=config.model.degrees,
                support=config.model.support,
                dropout_rate=getattr(config.training, 'dropout_rate', 0.1),
                use_spectral_norm=getattr(config.training, 'use_spectral_norm', False)
            ).to(device)
            binet.load_state_dict(model_state)
            binet.eval()
            
            with torch.no_grad():
                # Generate latent codes
                latent_codes = binet.encode(real_points)
                # Reconstruct
                reconstructed = binet.decode(latent_codes)
                model_predictions.append(reconstructed)
        
        # Average predictions (ensemble)
        ensemble_pred = torch.stack(model_predictions).mean(dim=0)
        ensemble_results.append(ensemble_pred)
    
    return ensemble_results


def save_best_fold_model(cv_results, config, save_path):
    """
    Save the best model from cross-validation.
    
    Args:
        cv_results: Results from k_fold_cross_validation
        config: Configuration object
        save_path: Path to save the best model
    """
    if cv_results['cv_stats'] is None:
        raise ValueError("No valid cross-validation results to save")
    
    best_fold_idx = cv_results['cv_stats']['best_fold'] - 1
    best_model_state = cv_results['best_models'][best_fold_idx]
    
    if best_model_state is None:
        raise ValueError("Best model state is None")
    
    # Save the best model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_model_state, save_path)
    
    return save_path