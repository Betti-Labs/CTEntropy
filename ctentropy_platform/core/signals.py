"""
Signal generation utilities for testing and validation.

This module provides the signal generators from the original CTEntropy research
for creating synthetic neurological signals representing different conditions.
"""

import numpy as np
from typing import Optional
from enum import Enum


class ConditionType(Enum):
    """Enumeration of neurological conditions for signal generation."""
    HEALTHY = "healthy"
    CTE = "cte"
    ALZHEIMERS = "alzheimers"
    DEPRESSION = "depression"


class SignalGenerator:
    """
    Generator for synthetic neurological signals based on CTEntropy research.
    
    Creates realistic signal patterns that mimic different neurological conditions
    for testing and validation purposes.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the signal generator.
        
        Args:
            random_seed: Optional seed for reproducible random number generation
        """
        self.random_seed = random_seed
        if random_seed is not None:
            self.rng = np.random.RandomState(random_seed)
        else:
            self.rng = np.random.RandomState()
    
    def generate_healthy_series(self, length: int = 1000) -> np.ndarray:
        """
        Generate a healthy brain signal with stable entropy characteristics.
        
        Args:
            length: Length of the signal to generate
            
        Returns:
            Synthetic healthy brain signal
        """
        t = np.linspace(0, 10, length)
        
        # Base signal with multiple frequency components
        signal = (np.sin(2 * np.pi * 5 * t) + 
                 0.3 * np.sin(2 * np.pi * 20 * t))
        
        # Add moderate noise
        noise = self.rng.normal(0, 0.2, length)
        
        return signal + noise
    
    def generate_cte_like_series(self, length: int = 1000) -> np.ndarray:
        """
        Generate a CTE-like signal showing rapid collapse and noise disruption.
        
        Args:
            length: Length of the signal to generate
            
        Returns:
            Synthetic CTE-like brain signal
        """
        t = np.linspace(0, 10, length)
        
        # Base signal similar to healthy
        base = (np.sin(2 * np.pi * 5 * t) + 
                0.3 * np.sin(2 * np.pi * 20 * t))
        
        # Progressive noise increase (trauma-related disruption)
        noise = self.rng.normal(0, 0.5 * (t / t.max()), length)
        
        # Exponential decay (neural degradation)
        decay = np.exp(-0.2 * t)
        
        return (base * decay) + noise
    
    def generate_alzheimers_series(self, length: int = 1000) -> np.ndarray:
        """
        Generate an Alzheimer's-like signal showing slow, steady decay.
        
        Args:
            length: Length of the signal to generate
            
        Returns:
            Synthetic Alzheimer's-like brain signal
        """
        t = np.linspace(0, 10, length)
        
        # Base signal with slightly different frequency profile
        base = (np.sin(2 * np.pi * 5 * t) + 
                0.3 * np.sin(2 * np.pi * 15 * t))
        
        # Moderate noise
        noise = self.rng.normal(0, 0.4, length)
        
        # Linear decay (gradual cognitive decline)
        decay = np.linspace(1, 0.5, length)
        
        return (base * decay) + noise
    
    def generate_depression_series(self, length: int = 1000) -> np.ndarray:
        """
        Generate a depression-like signal showing symbolic stagnation.
        
        Args:
            length: Length of the signal to generate
            
        Returns:
            Synthetic depression-like brain signal
        """
        t = np.linspace(0, 10, length)
        
        # Lower frequency base (reduced activity)
        base = np.sin(2 * np.pi * 3 * t)
        
        # Repetitive locked loop pattern (rumination/stagnation)
        locked_loop = np.tile(np.sin(2 * np.pi * 0.5 * t[:100]), 10)
        
        # Low noise (reduced variability)
        noise = self.rng.normal(0, 0.1, length)
        
        return base + 0.3 * locked_loop[:length] + noise
    
    def generate_signal(self, condition: ConditionType, length: int = 1000) -> np.ndarray:
        """
        Generate a signal for the specified condition type.
        
        Args:
            condition: Type of neurological condition to simulate
            length: Length of the signal to generate
            
        Returns:
            Synthetic brain signal for the specified condition
        """
        if condition == ConditionType.HEALTHY:
            return self.generate_healthy_series(length)
        elif condition == ConditionType.CTE:
            return self.generate_cte_like_series(length)
        elif condition == ConditionType.ALZHEIMERS:
            return self.generate_alzheimers_series(length)
        elif condition == ConditionType.DEPRESSION:
            return self.generate_depression_series(length)
        else:
            raise ValueError(f"Unknown condition type: {condition}")
    
    def generate_batch(self, conditions: list, length: int = 1000, 
                      samples_per_condition: int = 1) -> dict:
        """
        Generate multiple signals for batch processing.
        
        Args:
            conditions: List of condition types to generate
            length: Length of each signal
            samples_per_condition: Number of samples to generate per condition
            
        Returns:
            Dictionary mapping condition names to lists of signals
        """
        batch = {}
        
        for condition in conditions:
            if isinstance(condition, str):
                condition = ConditionType(condition)
                
            signals = []
            for _ in range(samples_per_condition):
                signal = self.generate_signal(condition, length)
                signals.append(signal)
                
            batch[condition.value] = signals
            
        return batch


# Legacy functions for backward compatibility
def generate_healthy_series(length: int = 1000) -> np.ndarray:
    """Legacy function from original CTEntropy research."""
    generator = SignalGenerator()
    return generator.generate_healthy_series(length)


def generate_cte_like_series(length: int = 1000) -> np.ndarray:
    """Legacy function from original CTEntropy research."""
    generator = SignalGenerator()
    return generator.generate_cte_like_series(length)


def generate_alzheimers_series(length: int = 1000) -> np.ndarray:
    """Legacy function from original CTEntropy research."""
    generator = SignalGenerator()
    return generator.generate_alzheimers_series(length)


def generate_depression_series(length: int = 1000) -> np.ndarray:
    """Legacy function from original CTEntropy research."""
    generator = SignalGenerator()
    return generator.generate_depression_series(length)