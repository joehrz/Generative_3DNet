# src/configs/config.py

import yaml
from typing import Any, Dict
from src.utils.exceptions import ConfigLoadError, ConfigValidationError
from src.configs.schema import ConfigSchema

class Config:
    def __init__(self, config_file: str):
        try:
            with open(config_file, 'r') as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigLoadError(f"Configuration file not found: {config_file}")
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Invalid YAML in configuration file {config_file}: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Failed to load configuration file {config_file}: {e}")
        
        if not isinstance(cfg, dict):
            raise ConfigValidationError(f"Configuration file must contain a dictionary, got {type(cfg)}")
        
        # Validate configuration against schema
        try:
            cfg = ConfigSchema.validate_full_config(cfg)
        except ConfigValidationError:
            raise
        except Exception as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")
        
        try:
            self.data = ConfigDict(cfg.get('data', {}))
            self.preprocessing = ConfigDict(cfg.get('preprocessing', {}))
            self.training = ConfigDict(cfg.get('training', {}))
            self.model = ConfigDict(cfg.get('model', {}))
        except Exception as e:
            raise ConfigValidationError(f"Failed to parse configuration sections: {e}")

class ConfigDict:
    def __init__(self, dictionary: Dict[str, Any]):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = ConfigDict(value)
            setattr(self, key, value)