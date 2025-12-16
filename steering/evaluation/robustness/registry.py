"""
Benchmark registry for robustness evaluation.

This module provides a registry pattern for discovering and instantiating
benchmark implementations.
"""

from typing import Dict, List, Type, Optional

from .base import BaseBenchmark


class BenchmarkRegistry:
    """Registry for robustness benchmarks.
    
    Provides a centralized way to register and retrieve benchmark implementations.
    
    Example:
        >>> @BenchmarkRegistry.register
        ... class MyBenchmark(BaseBenchmark):
        ...     pass
        
        >>> benchmark = BenchmarkRegistry.get("my_benchmark")
    """
    
    _benchmarks: Dict[str, Type[BaseBenchmark]] = {}
    
    @classmethod
    def register(cls, benchmark_class: Type[BaseBenchmark]) -> Type[BaseBenchmark]:
        """Register a benchmark class.
        
        Can be used as a decorator:
            @BenchmarkRegistry.register
            class MyBenchmark(BaseBenchmark):
                ...
        
        Args:
            benchmark_class: The benchmark class to register.
            
        Returns:
            The registered benchmark class (allows use as decorator).
            
        Raises:
            ValueError: If a benchmark with the same name is already registered.
        """
        # Instantiate temporarily to get the name
        instance = benchmark_class()
        name = instance.name
        
        if name in cls._benchmarks:
            raise ValueError(f"Benchmark '{name}' is already registered")
        
        cls._benchmarks[name] = benchmark_class
        return benchmark_class
    
    @classmethod
    def get(cls, name: str) -> BaseBenchmark:
        """Get a benchmark instance by name.
        
        Args:
            name: The benchmark name.
            
        Returns:
            A new instance of the benchmark.
            
        Raises:
            ValueError: If the benchmark is not found.
        """
        if name not in cls._benchmarks:
            available = cls.list_benchmarks()
            raise ValueError(
                f"Benchmark '{name}' not found. Available: {available}"
            )
        
        return cls._benchmarks[name]()
    
    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all registered benchmark names.
        
        Returns:
            List of registered benchmark names.
        """
        return list(cls._benchmarks.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a benchmark is registered.
        
        Args:
            name: The benchmark name to check.
            
        Returns:
            True if the benchmark is registered, False otherwise.
        """
        return name in cls._benchmarks
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered benchmarks.
        
        Mainly useful for testing.
        """
        cls._benchmarks.clear()
