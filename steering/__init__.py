"""
Selective Steering Core Package

This package contains the core components for implementing Selective Steering
for Large Language Models.
"""

from .data import DataManager
from .extraction import ActivationExtractor
from .direction import FeatureDirectionCalculator
from .plane import SteeringPlaneConstructor
from .steering import (
    AngularSteeringOperator,
    AdaptiveSteeringOperator,
    SelectiveSteeringOperator,
    AdditionSteeringOperator,
    AblationSteeringOperator,
)
from .hooks import ModelHookManager
from .evaluation import EvaluationSuite
from .pipeline import AngularSteeringPipeline
from .artifacts import ArtifactsManager, ActivationAnalyzer

# vLLM steering integration
from .vllm_steering import (
    SteeringLLM,
    create_steering_hook,
)

__version__ = "0.1.0"

__all__ = [
    # Core components
    "DataManager",
    "ActivationExtractor",
    "FeatureDirectionCalculator",
    "SteeringPlaneConstructor",
    "AngularSteeringOperator",
    "AdaptiveSteeringOperator",
    "SelectiveSteeringOperator",
    "AdditionSteeringOperator",
    "AblationSteeringOperator",
    "ModelHookManager",
    "EvaluationSuite",
    "AngularSteeringPipeline",
    "ArtifactsManager",
    "ActivationAnalyzer",
    # vLLM steering
    "SteeringLLM",
    "create_steering_hook",
]