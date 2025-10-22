"""Configuration loading utilities"""

import yaml
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Load and merge configuration files"""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """
        Merge two configuration dictionaries
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def save(config: Dict[str, Any], path: str) -> None:
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary
            path: Output path
        """
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)