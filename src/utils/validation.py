# File: src/utils/validation.py

"""
Comprehensive input validation utilities for the BI-Net project.
Provides validation functions for point clouds, tensors, and configuration parameters.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional, Any, Dict, List
import warnings


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_point_cloud_tensor(
    pc: torch.Tensor,
    name: str = "point_cloud",
    min_points: int = 1,
    max_points: int = 100000,
    expected_dim: int = 3,
    allow_batch: bool = True
) -> torch.Tensor:
    """
    Validate a point cloud tensor with comprehensive checks.
    
    Args:
        pc: Point cloud tensor
        name: Name for error messages
        min_points: Minimum number of points
        max_points: Maximum number of points
        expected_dim: Expected feature dimension (usually 3 for XYZ)
        allow_batch: Whether to allow batch dimension
    
    Returns:
        torch.Tensor: Validated point cloud tensor
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(pc, torch.Tensor):
        raise ValidationError(f"{name} must be a torch.Tensor, got {type(pc)}")
    
    if pc.numel() == 0:
        raise ValidationError(f"{name} cannot be empty")
    
    # Check dimensions
    if allow_batch:
        if pc.dim() not in [2, 3]:
            raise ValidationError(f"{name} must be 2D (N, {expected_dim}) or 3D (B, N, {expected_dim}), got shape {pc.shape}")
        
        if pc.dim() == 3:
            batch_size, num_points, feat_dim = pc.shape
            if batch_size == 0:
                raise ValidationError(f"{name} batch size cannot be 0")
        else:
            num_points, feat_dim = pc.shape
            
    else:
        if pc.dim() != 2:
            raise ValidationError(f"{name} must be 2D (N, {expected_dim}), got shape {pc.shape}")
        num_points, feat_dim = pc.shape
    
    # Validate feature dimension
    if feat_dim != expected_dim:
        raise ValidationError(f"{name} feature dimension must be {expected_dim}, got {feat_dim}")
    
    # Validate number of points
    if num_points < min_points:
        raise ValidationError(f"{name} must have at least {min_points} points, got {num_points}")
    
    if num_points > max_points:
        raise ValidationError(f"{name} cannot have more than {max_points} points, got {num_points}")
    
    # Check for NaN or infinite values
    if torch.any(torch.isnan(pc)):
        raise ValidationError(f"{name} contains NaN values")
    
    if torch.any(torch.isinf(pc)):
        raise ValidationError(f"{name} contains infinite values")
    
    # Check for reasonable coordinate ranges
    if torch.any(torch.abs(pc) > 1000):
        warnings.warn(f"{name} contains very large coordinates (>1000), this might cause numerical issues")
    
    return pc


def validate_point_cloud_numpy(
    pc: np.ndarray,
    name: str = "point_cloud",
    min_points: int = 1,
    max_points: int = 100000,
    expected_dim: int = 3
) -> np.ndarray:
    """
    Validate a point cloud numpy array.
    
    Args:
        pc: Point cloud numpy array
        name: Name for error messages
        min_points: Minimum number of points
        max_points: Maximum number of points
        expected_dim: Expected feature dimension
    
    Returns:
        np.ndarray: Validated point cloud array
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(pc, np.ndarray):
        raise ValidationError(f"{name} must be a numpy.ndarray, got {type(pc)}")
    
    if pc.size == 0:
        raise ValidationError(f"{name} cannot be empty")
    
    if pc.ndim != 2:
        raise ValidationError(f"{name} must be 2D (N, {expected_dim}), got shape {pc.shape}")
    
    num_points, feat_dim = pc.shape
    
    if feat_dim != expected_dim:
        raise ValidationError(f"{name} feature dimension must be {expected_dim}, got {feat_dim}")
    
    if num_points < min_points:
        raise ValidationError(f"{name} must have at least {min_points} points, got {num_points}")
    
    if num_points > max_points:
        raise ValidationError(f"{name} cannot have more than {max_points} points, got {num_points}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(pc)):
        raise ValidationError(f"{name} contains NaN values")
    
    if np.any(np.isinf(pc)):
        raise ValidationError(f"{name} contains infinite values")
    
    # Check data type
    if not np.issubdtype(pc.dtype, np.floating):
        warnings.warn(f"{name} is not a floating point array, converting to float32")
        pc = pc.astype(np.float32)
    
    return pc


def validate_latent_code(
    latent: torch.Tensor,
    name: str = "latent_code",
    expected_dim: int = 128,
    allow_batch: bool = True
) -> torch.Tensor:
    """
    Validate a latent code tensor.
    
    Args:
        latent: Latent code tensor
        name: Name for error messages
        expected_dim: Expected latent dimension
        allow_batch: Whether to allow batch dimension
    
    Returns:
        torch.Tensor: Validated latent code tensor
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(latent, torch.Tensor):
        raise ValidationError(f"{name} must be a torch.Tensor, got {type(latent)}")
    
    if latent.numel() == 0:
        raise ValidationError(f"{name} cannot be empty")
    
    if allow_batch:
        if latent.dim() not in [1, 2]:
            raise ValidationError(f"{name} must be 1D ({expected_dim},) or 2D (B, {expected_dim}), got shape {latent.shape}")
        
        if latent.dim() == 2:
            batch_size, latent_dim = latent.shape
            if batch_size == 0:
                raise ValidationError(f"{name} batch size cannot be 0")
        else:
            latent_dim = latent.shape[0]
    else:
        if latent.dim() != 1:
            raise ValidationError(f"{name} must be 1D ({expected_dim},), got shape {latent.shape}")
        latent_dim = latent.shape[0]
    
    if latent_dim != expected_dim:
        raise ValidationError(f"{name} dimension must be {expected_dim}, got {latent_dim}")
    
    # Check for NaN or infinite values
    if torch.any(torch.isnan(latent)):
        raise ValidationError(f"{name} contains NaN values")
    
    if torch.any(torch.isinf(latent)):
        raise ValidationError(f"{name} contains infinite values")
    
    return latent


def validate_config_parameters(config: Dict[str, Any], name: str = "config") -> Dict[str, Any]:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        name: Name for error messages
    
    Returns:
        Dict[str, Any]: Validated configuration
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(config, dict):
        raise ValidationError(f"{name} must be a dictionary, got {type(config)}")
    
    # Check for required sections
    required_sections = ['data', 'model', 'training']
    for section in required_sections:
        if section not in config:
            raise ValidationError(f"{name} missing required section: {section}")
    
    # Validate data section
    data_config = config['data']
    required_data_keys = ['raw_dir', 'processed_dir', 'splits_dir']
    for key in required_data_keys:
        if key not in data_config:
            raise ValidationError(f"{name}.data missing required key: {key}")
    
    # Validate model section
    model_config = config['model']
    required_model_keys = ['latent_dim', 'save_dir']
    for key in required_model_keys:
        if key not in model_config:
            raise ValidationError(f"{name}.model missing required key: {key}")
    
    # Validate training section
    training_config = config['training']
    required_training_keys = ['batch_size', 'learning_rate', 'epochs', 'device']
    for key in required_training_keys:
        if key not in training_config:
            raise ValidationError(f"{name}.training missing required key: {key}")
    
    # Validate value ranges
    if model_config['latent_dim'] <= 0:
        raise ValidationError(f"{name}.model.latent_dim must be positive, got {model_config['latent_dim']}")
    
    if training_config['batch_size'] <= 0:
        raise ValidationError(f"{name}.training.batch_size must be positive, got {training_config['batch_size']}")
    
    if training_config['learning_rate'] <= 0:
        raise ValidationError(f"{name}.training.learning_rate must be positive, got {training_config['learning_rate']}")
    
    if training_config['epochs'] <= 0:
        raise ValidationError(f"{name}.training.epochs must be positive, got {training_config['epochs']}")
    
    return config


def validate_device(device: Union[str, torch.device], name: str = "device") -> torch.device:
    """
    Validate and convert device specification.
    
    Args:
        device: Device specification
        name: Name for error messages
    
    Returns:
        torch.device: Validated device
    
    Raises:
        ValidationError: If validation fails
    """
    if isinstance(device, str):
        try:
            device = torch.device(device)
        except RuntimeError as e:
            raise ValidationError(f"Invalid {name} specification: {e}")
    elif not isinstance(device, torch.device):
        raise ValidationError(f"{name} must be a string or torch.device, got {type(device)}")
    
    # Check if CUDA device is available if requested
    if device.type == 'cuda' and not torch.cuda.is_available():
        warnings.warn(f"CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    
    return device


def validate_file_path(path: str, name: str = "file_path", check_exists: bool = True) -> str:
    """
    Validate file path.
    
    Args:
        path: File path to validate
        name: Name for error messages
        check_exists: Whether to check if file exists
    
    Returns:
        str: Validated file path
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(path, str):
        raise ValidationError(f"{name} must be a string, got {type(path)}")
    
    if not path.strip():
        raise ValidationError(f"{name} cannot be empty")
    
    if check_exists:
        import os
        if not os.path.exists(path):
            raise ValidationError(f"{name} does not exist: {path}")
    
    return path


def validate_batch_size(batch_size: int, name: str = "batch_size") -> int:
    """
    Validate batch size parameter.
    
    Args:
        batch_size: Batch size to validate
        name: Name for error messages
    
    Returns:
        int: Validated batch size
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(batch_size, int):
        raise ValidationError(f"{name} must be an integer, got {type(batch_size)}")
    
    if batch_size <= 0:
        raise ValidationError(f"{name} must be positive, got {batch_size}")
    
    if batch_size > 1024:
        warnings.warn(f"{name} is very large ({batch_size}), this might cause memory issues")
    
    return batch_size


def validate_learning_rate(lr: float, name: str = "learning_rate") -> float:
    """
    Validate learning rate parameter.
    
    Args:
        lr: Learning rate to validate
        name: Name for error messages
    
    Returns:
        float: Validated learning rate
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(lr, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(lr)}")
    
    if lr <= 0:
        raise ValidationError(f"{name} must be positive, got {lr}")
    
    if lr > 1.0:
        warnings.warn(f"{name} is very large ({lr}), this might cause training instability")
    
    if lr < 1e-6:
        warnings.warn(f"{name} is very small ({lr}), this might cause slow training")
    
    return float(lr)


def validate_augmentation_parameters(
    aug_params: Dict[str, Any],
    name: str = "augmentation_params"
) -> Dict[str, Any]:
    """
    Validate augmentation parameters.
    
    Args:
        aug_params: Augmentation parameters dictionary
        name: Name for error messages
    
    Returns:
        Dict[str, Any]: Validated augmentation parameters
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(aug_params, dict):
        raise ValidationError(f"{name} must be a dictionary, got {type(aug_params)}")
    
    # Validate probability parameters
    for key, value in aug_params.items():
        if key.endswith('_prob') or key.startswith('p_'):
            if not isinstance(value, (int, float)):
                raise ValidationError(f"{name}.{key} must be a number, got {type(value)}")
            if not (0 <= value <= 1):
                raise ValidationError(f"{name}.{key} must be in [0, 1], got {value}")
    
    # Validate specific parameters
    if 'noise_std' in aug_params:
        if aug_params['noise_std'] < 0:
            raise ValidationError(f"{name}.noise_std must be non-negative, got {aug_params['noise_std']}")
    
    if 'jitter_sigma' in aug_params:
        if aug_params['jitter_sigma'] < 0:
            raise ValidationError(f"{name}.jitter_sigma must be non-negative, got {aug_params['jitter_sigma']}")
    
    return aug_params