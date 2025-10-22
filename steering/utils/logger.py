"""Logging utilities"""

import logging
import sys
import inspect
from pathlib import Path
from datetime import datetime
from typing import Optional, Any


def get_dynamic_logger_name(obj: Any) -> str:
    """
    Generate dynamic logger name based on object's module and class

    Args:
        obj: Object instance to generate logger name for

    Returns:
        Logger name in format "steering.{module_name}.{class_name}"
    """
    # Get the class name
    class_name = obj.__class__.__name__

    # Get the module name from the object's module
    module = inspect.getmodule(obj.__class__)
    if module and module.__name__:
        # Extract module path relative to steering package
        module_parts = module.__name__.split('.')

        # Find steering package index
        try:
            steering_idx = module_parts.index('steering')
            # Get parts after 'steering'
            if len(module_parts) > steering_idx + 1:
                # Get the immediate submodule (e.g., 'pipeline', 'data', etc.)
                submodule = module_parts[steering_idx + 1]
                # Get the filename (last part before class)
                if len(module_parts) > steering_idx + 2:
                    filename = module_parts[-1]  # Last part is usually the filename
                    return f"steering.{submodule}.{filename}.{class_name}"
                else:
                    return f"steering.{submodule}.{class_name}"
            else:
                return f"steering.{class_name}"
        except ValueError:
            # If 'steering' not found in module path, use full module name
            return f"{module.__name__}.{class_name}"

    # Fallback to just class name
    return f"steering.{class_name}"


def setup_logger(
    name: Optional[str] = None,
    obj: Optional[Any] = None,
    log_dir: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name (if not provided, will be generated from obj)
        obj: Object instance to generate dynamic logger name from
        log_dir: Directory for log files
        level: Logging level

    Returns:
        Configured logger
    """
    # Generate dynamic name if obj provided and name not specified
    if name is None and obj is not None:
        name = get_dynamic_logger_name(obj)
    elif name is None:
        name = "steering"
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(log_dir) / f'steering_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class SteeringLogger:
    """Logger for steering operations"""

    def __init__(self, log_dir: str = "logs"):
        self.logger = setup_logger(obj=self, log_dir=log_dir)
    
    def log_calibration(self, info: dict):
        """Log calibration information"""
        self.logger.info("Calibration completed")
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_steering(self, theta: float, metrics: dict):
        """Log steering results"""
        self.logger.info(f"Steering at θ={theta}°")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
    
    def log_error(self, error: Exception, context: str):
        """Log error with context"""
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)