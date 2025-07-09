# File: src/utils/exceptions.py

"""
Custom exception classes for the BI-Net project.
Provides specific exception types for better error handling and debugging.
"""


class BiNetError(Exception):
    """Base exception class for BI-Net related errors."""
    pass


class DataError(BiNetError):
    """Exception raised for data-related errors."""
    pass


class ModelError(BiNetError):
    """Exception raised for model-related errors."""
    pass


class TrainingError(BiNetError):
    """Exception raised for training-related errors."""
    pass


class ConfigurationError(BiNetError):
    """Exception raised for configuration-related errors."""
    pass


class PreprocessingError(DataError):
    """Exception raised during data preprocessing."""
    pass


class DatasetError(DataError):
    """Exception raised for dataset loading/handling errors."""
    pass


class PointCloudError(DataError):
    """Exception raised for point cloud specific errors."""
    pass


class ModelArchitectureError(ModelError):
    """Exception raised for model architecture issues."""
    pass


class ModelLoadError(ModelError):
    """Exception raised when model loading fails."""
    pass


class ModelSaveError(ModelError):
    """Exception raised when model saving fails."""
    pass


class TrainingLoopError(TrainingError):
    """Exception raised during training loop execution."""
    pass


class OptimizationError(TrainingError):
    """Exception raised for optimization-related issues."""
    pass


class LossComputationError(TrainingError):
    """Exception raised when loss computation fails."""
    pass


class ValidationError(TrainingError):
    """Exception raised during validation process."""
    pass


class ConfigValidationError(ConfigurationError):
    """Exception raised for configuration validation errors."""
    pass


class ConfigLoadError(ConfigurationError):
    """Exception raised when configuration loading fails."""
    pass


class DeviceError(BiNetError):
    """Exception raised for device-related errors."""
    pass


class GPUError(DeviceError):
    """Exception raised for GPU-specific errors."""
    pass


class MemoryError(BiNetError):
    """Exception raised for memory-related errors."""
    pass


class AugmentationError(DataError):
    """Exception raised during data augmentation."""
    pass


class EMDError(BiNetError):
    """Exception raised for EMD computation errors."""
    pass


class CrossValidationError(TrainingError):
    """Exception raised during cross-validation."""
    pass


class FileIOError(BiNetError):
    """Exception raised for file I/O operations."""
    pass


class NetworkError(BiNetError):
    """Exception raised for network-related operations."""
    pass


class VisualizationError(BiNetError):
    """Exception raised for visualization-related errors."""
    pass