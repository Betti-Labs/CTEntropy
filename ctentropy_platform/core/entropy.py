"""
Core symbolic entropy calculation engine based on CTEntropy research.

This module implements the symbolic entropy function that analyzes spectral
distribution and decay patterns in neurological signals.
"""

import numpy as np
from scipy.fftpack import fft
from typing import Optional, Tuple
import warnings


class SymbolicEntropyCalculator:
    """
    Symbolic entropy calculator using FFT-based spectral analysis.
    
    Based on the CTEntropy research methodology for detecting entropy collapse
    patterns in neurological signals.
    """
    
    def __init__(self, window_size: int = 50, overlap: float = 0.0):
        """
        Initialize the symbolic entropy calculator.
        
        Args:
            window_size: Size of the sliding window for entropy calculation
            overlap: Overlap fraction between windows (0.0 to 0.9)
        """
        self.window_size = window_size
        self.overlap = max(0.0, min(0.9, overlap))  # Clamp between 0 and 0.9
        self.step_size = int(window_size * (1 - overlap))
        
    def calculate(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate symbolic entropy for a signal using sliding windows.
        
        Args:
            signal: Input neurological signal (1D numpy array)
            
        Returns:
            Array of entropy values for each window
            
        Raises:
            ValueError: If signal is too short or invalid
        """
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
            
        if len(signal.shape) != 1:
            raise ValueError("Signal must be 1-dimensional")
            
        if len(signal) < self.window_size:
            raise ValueError(f"Signal length ({len(signal)}) must be >= window_size ({self.window_size})")
            
        entropies = []
        
        # Sliding window entropy calculation
        for i in range(0, len(signal) - self.window_size + 1, self.step_size):
            segment = signal[i:i + self.window_size]
            entropy = self._calculate_segment_entropy(segment)
            entropies.append(entropy)
            
        return np.array(entropies)
    
    def _calculate_segment_entropy(self, segment: np.ndarray) -> float:
        """
        Calculate entropy for a single signal segment.
        
        Args:
            segment: Signal segment to analyze
            
        Returns:
            Entropy value for the segment
        """
        # Apply FFT to get frequency spectrum
        spectrum = np.abs(fft(segment))[:self.window_size // 2]
        
        # Handle edge case of zero spectrum
        if np.sum(spectrum) == 0:
            warnings.warn("Zero spectrum detected, returning minimum entropy")
            return 0.0
            
        # Normalize spectrum to create probability distribution
        spectrum = spectrum / np.sum(spectrum)
        
        # Calculate Shannon entropy with numerical stability
        # Add small epsilon to prevent log(0)
        epsilon = 1e-9
        entropy = -np.sum(spectrum * np.log2(spectrum + epsilon))
        
        # Ensure entropy is non-negative (numerical precision issues)
        return max(0.0, entropy)
    
    def calculate_with_metadata(self, signal: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Calculate entropy with additional metadata about the calculation.
        
        Args:
            signal: Input neurological signal
            
        Returns:
            Tuple of (entropy_values, metadata_dict)
        """
        entropies = self.calculate(signal)
        
        metadata = {
            'window_size': self.window_size,
            'overlap': self.overlap,
            'step_size': self.step_size,
            'num_windows': len(entropies),
            'signal_length': len(signal),
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'entropy_range': (np.min(entropies), np.max(entropies))
        }
        
        return entropies, metadata
    
    def detect_entropy_collapse(self, entropies: np.ndarray, threshold: float = 0.1) -> dict:
        """
        Detect entropy collapse patterns in the calculated entropies.
        
        Args:
            entropies: Array of entropy values
            threshold: Threshold for detecting significant entropy drops
            
        Returns:
            Dictionary with collapse detection results
        """
        if len(entropies) < 2:
            return {'collapse_detected': False, 'reason': 'Insufficient data'}
            
        # Calculate entropy trend (linear regression slope)
        x = np.arange(len(entropies))
        slope = np.polyfit(x, entropies, 1)[0]
        
        # Calculate relative entropy drop
        initial_entropy = np.mean(entropies[:max(1, len(entropies)//4)])
        final_entropy = np.mean(entropies[-max(1, len(entropies)//4):])
        relative_drop = (initial_entropy - final_entropy) / initial_entropy if initial_entropy > 0 else 0
        
        # Detect collapse
        collapse_detected = bool((slope < -threshold) or (relative_drop > threshold))
        
        return {
            'collapse_detected': collapse_detected,
            'slope': slope,
            'relative_drop': relative_drop,
            'initial_entropy': initial_entropy,
            'final_entropy': final_entropy,
            'threshold_used': threshold
        }


def symbolic_entropy(signal: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Legacy function for backward compatibility with original CTEntropy research.
    
    Args:
        signal: Input signal
        window: Window size for entropy calculation
        
    Returns:
        Array of entropy values
    """
    calculator = SymbolicEntropyCalculator(window_size=window)
    return calculator.calculate(signal)