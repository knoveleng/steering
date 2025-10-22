"""
Angular Steering Core Package

This package contains the core components for implementing Angular Steering
for Large Language Models.
"""

from .data import DataManager
from .extraction import ActivationExtractor
from .direction import FeatureDirectionCalculator
from .plane import SteeringPlaneConstructor
from .steering import AngularSteeringOperator, AdaptiveSteeringOperator
from .hooks import ModelHookManager
from .evaluation import EvaluationSuite
from .pipeline import AngularSteeringPipeline
from .artifacts import ArtifactsManager, ActivationAnalyzer

__version__ = "0.1.0"

__all__ = [
    "DataManager",
    "ActivationExtractor",
    "FeatureDirectionCalculator",
    "SteeringPlaneConstructor",
    "AngularSteeringOperator",
    "AdaptiveSteeringOperator",
    "ModelHookManager",
    "EvaluationSuite",
    "AngularSteeringPipeline",
    "ArtifactsManager",
    "ActivationAnalyzer",
]