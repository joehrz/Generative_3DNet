# File: src/configs/schema.py

"""
Configuration schema validation for the BI-Net project.
Provides comprehensive validation of configuration parameters.
"""

import os
from typing import Dict, Any, List, Union
from src.utils.exceptions import ConfigValidationError


class ConfigSchema:
    """Configuration schema validator for BI-Net project."""
    
    @staticmethod
    def validate_data_config(data_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data configuration section.
        
        Args:
            data_config: Data configuration dictionary
            
        Returns:
            Dict[str, Any]: Validated data configuration
            
        Raises:
            ConfigValidationError: If validation fails
        """
        required_keys = ['raw_dir', 'processed_dir', 'splits_dir']
        
        for key in required_keys:
            if key not in data_config:
                raise ConfigValidationError(f"Missing required data config key: {key}")
            
            if not isinstance(data_config[key], str):
                raise ConfigValidationError(f"data.{key} must be a string, got {type(data_config[key])}")
            
            if not data_config[key].strip():
                raise ConfigValidationError(f"data.{key} cannot be empty")
        
        # Validate split ratios if present
        if 'split_ratios' in data_config:
            ratios = data_config['split_ratios']
            if not isinstance(ratios, list):
                raise ConfigValidationError(f"data.split_ratios must be a list, got {type(ratios)}")
            
            if len(ratios) != 3:
                raise ConfigValidationError(f"data.split_ratios must have exactly 3 values (train, val, test), got {len(ratios)}")
            
            for i, ratio in enumerate(ratios):
                if not isinstance(ratio, (int, float)):
                    raise ConfigValidationError(f"data.split_ratios[{i}] must be a number, got {type(ratio)}")
                if ratio < 0 or ratio > 1:
                    raise ConfigValidationError(f"data.split_ratios[{i}] must be in [0, 1], got {ratio}")
            
            if abs(sum(ratios) - 1.0) > 1e-6:
                raise ConfigValidationError(f"data.split_ratios must sum to 1.0, got {sum(ratios)}")
        
        return data_config
    
    @staticmethod
    def validate_preprocessing_config(prep_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate preprocessing configuration section.
        
        Args:
            prep_config: Preprocessing configuration dictionary
            
        Returns:
            Dict[str, Any]: Validated preprocessing configuration
            
        Raises:
            ConfigValidationError: If validation fails
        """
        # Validate voxel_size
        if 'voxel_size' in prep_config:
            voxel_size = prep_config['voxel_size']
            if not isinstance(voxel_size, (int, float)):
                raise ConfigValidationError(f"preprocessing.voxel_size must be a number, got {type(voxel_size)}")
            if voxel_size <= 0:
                raise ConfigValidationError(f"preprocessing.voxel_size must be positive, got {voxel_size}")
        
        # Validate num_points
        if 'num_points' in prep_config:
            num_points = prep_config['num_points']
            if not isinstance(num_points, int):
                raise ConfigValidationError(f"preprocessing.num_points must be an integer, got {type(num_points)}")
            if num_points <= 0:
                raise ConfigValidationError(f"preprocessing.num_points must be positive, got {num_points}")
            if num_points > 100000:
                raise ConfigValidationError(f"preprocessing.num_points is too large (max 100000), got {num_points}")
        
        # Validate boolean flags
        bool_flags = ['use_fps', 'skip_downsample', 'normalize']
        for flag in bool_flags:
            if flag in prep_config:
                if not isinstance(prep_config[flag], bool):
                    raise ConfigValidationError(f"preprocessing.{flag} must be a boolean, got {type(prep_config[flag])}")
        
        return prep_config
    
    @staticmethod
    def validate_model_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model configuration section.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Dict[str, Any]: Validated model configuration
            
        Raises:
            ConfigValidationError: If validation fails
        """
        required_keys = ['latent_dim', 'save_dir']
        
        for key in required_keys:
            if key not in model_config:
                raise ConfigValidationError(f"Missing required model config key: {key}")
        
        # Validate latent_dim
        latent_dim = model_config['latent_dim']
        if not isinstance(latent_dim, int):
            raise ConfigValidationError(f"model.latent_dim must be an integer, got {type(latent_dim)}")
        if latent_dim <= 0:
            raise ConfigValidationError(f"model.latent_dim must be positive, got {latent_dim}")
        if latent_dim > 1024:
            raise ConfigValidationError(f"model.latent_dim is too large (max 1024), got {latent_dim}")
        
        # Validate save_dir
        save_dir = model_config['save_dir']
        if not isinstance(save_dir, str):
            raise ConfigValidationError(f"model.save_dir must be a string, got {type(save_dir)}")
        if not save_dir.strip():
            raise ConfigValidationError(f"model.save_dir cannot be empty")
        
        # Validate TreeGCN parameters if present
        if 'features_g' in model_config:
            features = model_config['features_g']
            if not isinstance(features, list):
                raise ConfigValidationError(f"model.features_g must be a list, got {type(features)}")
            if len(features) < 2:
                raise ConfigValidationError(f"model.features_g must have at least 2 elements, got {len(features)}")
            for i, feat in enumerate(features):
                if not isinstance(feat, int):
                    raise ConfigValidationError(f"model.features_g[{i}] must be an integer, got {type(feat)}")
                if feat <= 0:
                    raise ConfigValidationError(f"model.features_g[{i}] must be positive, got {feat}")
        
        if 'degrees_g' in model_config:
            degrees = model_config['degrees_g']
            if not isinstance(degrees, list):
                raise ConfigValidationError(f"model.degrees_g must be a list, got {type(degrees)}")
            for i, deg in enumerate(degrees):
                if not isinstance(deg, int):
                    raise ConfigValidationError(f"model.degrees_g[{i}] must be an integer, got {type(deg)}")
                if deg <= 0:
                    raise ConfigValidationError(f"model.degrees_g[{i}] must be positive, got {deg}")
        
        # Validate dropout rate
        if 'dropout_rate' in model_config:
            dropout = model_config['dropout_rate']
            if not isinstance(dropout, (int, float)):
                raise ConfigValidationError(f"model.dropout_rate must be a number, got {type(dropout)}")
            if dropout < 0 or dropout > 1:
                raise ConfigValidationError(f"model.dropout_rate must be in [0, 1], got {dropout}")
        
        return model_config
    
    @staticmethod
    def validate_training_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate training configuration section.
        
        Args:
            training_config: Training configuration dictionary
            
        Returns:
            Dict[str, Any]: Validated training configuration
            
        Raises:
            ConfigValidationError: If validation fails
        """
        required_keys = ['batch_size', 'epochs', 'device']
        
        for key in required_keys:
            if key not in training_config:
                raise ConfigValidationError(f"Missing required training config key: {key}")
        
        # Validate batch_size
        batch_size = training_config['batch_size']
        if not isinstance(batch_size, int):
            raise ConfigValidationError(f"training.batch_size must be an integer, got {type(batch_size)}")
        if batch_size <= 0:
            raise ConfigValidationError(f"training.batch_size must be positive, got {batch_size}")
        if batch_size > 1024:
            raise ConfigValidationError(f"training.batch_size is too large (max 1024), got {batch_size}")
        
        # Validate learning rates (encoder, decoder, discriminator)
        lr_keys = ['lr_enc', 'lr_dec', 'lr_disc']
        for lr_key in lr_keys:
            if lr_key in training_config:
                lr = training_config[lr_key]
                if not isinstance(lr, (int, float)):
                    raise ConfigValidationError(f"training.{lr_key} must be a number, got {type(lr)}")
                if lr <= 0:
                    raise ConfigValidationError(f"training.{lr_key} must be positive, got {lr}")
                if lr > 1.0:
                    raise ConfigValidationError(f"training.{lr_key} is too large (max 1.0), got {lr}")
        
        # Validate beta values for optimizer
        if 'betas' in training_config:
            betas = training_config['betas']
            if not isinstance(betas, list):
                raise ConfigValidationError(f"training.betas must be a list, got {type(betas)}")
            if len(betas) != 2:
                raise ConfigValidationError(f"training.betas must have exactly 2 values, got {len(betas)}")
            for i, beta in enumerate(betas):
                if not isinstance(beta, (int, float)):
                    raise ConfigValidationError(f"training.betas[{i}] must be a number, got {type(beta)}")
                if beta < 0 or beta > 1:
                    raise ConfigValidationError(f"training.betas[{i}] must be in [0, 1], got {beta}")
        
        # Validate epochs
        epochs = training_config['epochs']
        if not isinstance(epochs, int):
            raise ConfigValidationError(f"training.epochs must be an integer, got {type(epochs)}")
        if epochs <= 0:
            raise ConfigValidationError(f"training.epochs must be positive, got {epochs}")
        if epochs > 10000:
            raise ConfigValidationError(f"training.epochs is too large (max 10000), got {epochs}")
        
        # Validate device
        device = training_config['device']
        if not isinstance(device, str):
            raise ConfigValidationError(f"training.device must be a string, got {type(device)}")
        valid_devices = ['cpu', 'cuda', 'auto']
        if device not in valid_devices and not device.startswith('cuda:'):
            raise ConfigValidationError(f"training.device must be one of {valid_devices} or 'cuda:X', got {device}")
        
        # Validate loss weights
        loss_weights = ['reconstruction_weight', 'adversarial_weight', 'uniformity_weight', 'gp_weight']
        for weight_name in loss_weights:
            if weight_name in training_config:
                weight = training_config[weight_name]
                if not isinstance(weight, (int, float)):
                    raise ConfigValidationError(f"training.{weight_name} must be a number, got {type(weight)}")
                if weight < 0:
                    raise ConfigValidationError(f"training.{weight_name} must be non-negative, got {weight}")
        
        # Validate warmup_epochs
        if 'warmup_epochs' in training_config:
            warmup = training_config['warmup_epochs']
            if not isinstance(warmup, int):
                raise ConfigValidationError(f"training.warmup_epochs must be an integer, got {type(warmup)}")
            if warmup < 0:
                raise ConfigValidationError(f"training.warmup_epochs must be non-negative, got {warmup}")
            if warmup >= epochs:
                raise ConfigValidationError(f"training.warmup_epochs ({warmup}) must be less than epochs ({epochs})")
        
        # Validate augmentation settings
        if 'augmentation' in training_config:
            aug_config = training_config['augmentation']
            if not isinstance(aug_config, dict):
                raise ConfigValidationError(f"training.augmentation must be a dictionary, got {type(aug_config)}")
            
            # Validate probability values
            for key, value in aug_config.items():
                if key.endswith('_prob') or key.startswith('p_'):
                    if not isinstance(value, (int, float)):
                        raise ConfigValidationError(f"training.augmentation.{key} must be a number, got {type(value)}")
                    if value < 0 or value > 1:
                        raise ConfigValidationError(f"training.augmentation.{key} must be in [0, 1], got {value}")
        
        return training_config
    
    @staticmethod
    def validate_full_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the complete configuration.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Dict[str, Any]: Validated configuration
            
        Raises:
            ConfigValidationError: If validation fails
        """
        if not isinstance(config, dict):
            raise ConfigValidationError(f"Configuration must be a dictionary, got {type(config)}")
        
        # Validate required top-level sections
        required_sections = ['data', 'training', 'model']
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError(f"Missing required configuration section: {section}")
        
        # Validate each section
        validated_config = {}
        
        validated_config['data'] = ConfigSchema.validate_data_config(config['data'])
        validated_config['training'] = ConfigSchema.validate_training_config(config['training'])
        validated_config['model'] = ConfigSchema.validate_model_config(config['model'])
        
        # Validate optional sections
        if 'preprocessing' in config:
            validated_config['preprocessing'] = ConfigSchema.validate_preprocessing_config(config['preprocessing'])
        
        # Copy other sections without validation
        for key, value in config.items():
            if key not in validated_config:
                validated_config[key] = value
        
        return validated_config