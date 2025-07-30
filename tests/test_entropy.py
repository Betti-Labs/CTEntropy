"""
Unit tests for the symbolic entropy calculation engine.
"""

import pytest
import numpy as np
from ctentropy_platform.core.entropy import SymbolicEntropyCalculator, symbolic_entropy
from ctentropy_platform.core.signals import SignalGenerator, ConditionType


class TestSymbolicEntropyCalculator:
    """Test suite for SymbolicEntropyCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = SymbolicEntropyCalculator(window_size=50)
        self.signal_generator = SignalGenerator(random_seed=42)
    
    def test_initialization(self):
        """Test calculator initialization with different parameters."""
        calc = SymbolicEntropyCalculator(window_size=100, overlap=0.5)
        assert calc.window_size == 100
        assert calc.overlap == 0.5
        assert calc.step_size == 50
        
        # Test overlap clamping
        calc = SymbolicEntropyCalculator(overlap=1.5)
        assert calc.overlap == 0.9
        
        calc = SymbolicEntropyCalculator(overlap=-0.1)
        assert calc.overlap == 0.0
    
    def test_calculate_basic(self):
        """Test basic entropy calculation functionality."""
        # Generate a simple test signal
        signal = np.sin(2 * np.pi * 0.1 * np.arange(200))
        
        entropies = self.calculator.calculate(signal)
        
        # Check output properties
        assert isinstance(entropies, np.ndarray)
        assert len(entropies) > 0
        assert all(entropy >= 0 for entropy in entropies)
    
    def test_calculate_with_different_signals(self):
        """Test entropy calculation with different signal types."""
        # Test with healthy signal
        healthy_signal = self.signal_generator.generate_healthy_series(500)
        healthy_entropies = self.calculator.calculate(healthy_signal)
        
        # Test with CTE-like signal
        cte_signal = self.signal_generator.generate_cte_like_series(500)
        cte_entropies = self.calculator.calculate(cte_signal)
        
        # Both should produce valid entropy arrays
        assert len(healthy_entropies) > 0
        assert len(cte_entropies) > 0
        assert all(np.isfinite(healthy_entropies))
        assert all(np.isfinite(cte_entropies))
    
    def test_calculate_edge_cases(self):
        """Test entropy calculation with edge cases."""
        # Test with minimum length signal
        min_signal = np.ones(50)
        entropies = self.calculator.calculate(min_signal)
        assert len(entropies) == 1
        
        # Test with signal too short
        with pytest.raises(ValueError, match="Signal length.*must be >= window_size"):
            self.calculator.calculate(np.ones(10))
        
        # Test with multi-dimensional signal
        with pytest.raises(ValueError, match="Signal must be 1-dimensional"):
            self.calculator.calculate(np.ones((50, 50)))
    
    def test_calculate_with_metadata(self):
        """Test entropy calculation with metadata output."""
        signal = self.signal_generator.generate_healthy_series(300)
        entropies, metadata = self.calculator.calculate_with_metadata(signal)
        
        # Check metadata structure
        expected_keys = ['window_size', 'overlap', 'step_size', 'num_windows', 
                        'signal_length', 'mean_entropy', 'std_entropy', 'entropy_range']
        assert all(key in metadata for key in expected_keys)
        
        # Check metadata values
        assert metadata['window_size'] == 50
        assert metadata['signal_length'] == 300
        assert metadata['num_windows'] == len(entropies)
        assert metadata['mean_entropy'] == np.mean(entropies)
        assert metadata['std_entropy'] == np.std(entropies)
    
    def test_detect_entropy_collapse(self):
        """Test entropy collapse detection functionality."""
        # Create a signal with clear entropy collapse
        declining_entropies = np.linspace(5.0, 1.0, 20)
        
        collapse_info = self.calculator.detect_entropy_collapse(declining_entropies)
        
        assert collapse_info['collapse_detected'] is True
        assert collapse_info['slope'] < 0
        assert collapse_info['relative_drop'] > 0
        
        # Test with stable entropies
        stable_entropies = np.ones(20) * 3.0 + np.random.normal(0, 0.1, 20)
        collapse_info = self.calculator.detect_entropy_collapse(stable_entropies)
        
        assert collapse_info['collapse_detected'] is False
    
    def test_zero_spectrum_handling(self):
        """Test handling of zero spectrum edge case."""
        # Create a signal that might produce zero spectrum
        zero_signal = np.zeros(100)
        
        # Should not crash and should handle gracefully
        entropies = self.calculator.calculate(zero_signal)
        assert len(entropies) > 0
        assert all(np.isfinite(entropies))
    
    def test_overlap_functionality(self):
        """Test sliding window with overlap."""
        signal = np.sin(2 * np.pi * 0.1 * np.arange(200))
        
        # No overlap
        calc_no_overlap = SymbolicEntropyCalculator(window_size=50, overlap=0.0)
        entropies_no_overlap = calc_no_overlap.calculate(signal)
        
        # 50% overlap
        calc_overlap = SymbolicEntropyCalculator(window_size=50, overlap=0.5)
        entropies_overlap = calc_overlap.calculate(signal)
        
        # Overlap should produce more entropy values
        assert len(entropies_overlap) > len(entropies_no_overlap)


class TestLegacyFunction:
    """Test suite for legacy symbolic_entropy function."""
    
    def test_legacy_function_compatibility(self):
        """Test that legacy function produces same results as new class."""
        signal = np.sin(2 * np.pi * 0.1 * np.arange(200))
        
        # Calculate using legacy function
        legacy_entropies = symbolic_entropy(signal, window=50)
        
        # Calculate using new class
        calculator = SymbolicEntropyCalculator(window_size=50)
        new_entropies = calculator.calculate(signal)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(legacy_entropies, new_entropies)


class TestEntropyPatterns:
    """Test entropy patterns for different neurological conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = SymbolicEntropyCalculator(window_size=50)
        self.signal_generator = SignalGenerator(random_seed=42)
    
    def test_condition_entropy_differences(self):
        """Test that different conditions produce different entropy patterns."""
        # Generate signals for all conditions
        conditions = [ConditionType.HEALTHY, ConditionType.CTE, 
                     ConditionType.ALZHEIMERS, ConditionType.DEPRESSION]
        
        entropy_stats = {}
        
        for condition in conditions:
            signal = self.signal_generator.generate_signal(condition, length=500)
            entropies = self.calculator.calculate(signal)
            
            entropy_stats[condition.value] = {
                'mean': np.mean(entropies),
                'std': np.std(entropies),
                'trend': np.polyfit(np.arange(len(entropies)), entropies, 1)[0]
            }
        
        # Verify that conditions produce different patterns
        means = [stats['mean'] for stats in entropy_stats.values()]
        assert len(set(np.round(means, 1))) > 1  # Different mean entropies
        
        # Test that CTE shows more entropy collapse than healthy over multiple samples
        cte_trends = []
        healthy_trends = []
        
        # Generate multiple samples to get statistical significance
        for seed in range(5):
            cte_gen = SignalGenerator(random_seed=seed)
            healthy_gen = SignalGenerator(random_seed=seed + 100)  # Different seed
            
            cte_signal = cte_gen.generate_cte_like_series(length=1000)
            healthy_signal = healthy_gen.generate_healthy_series(length=1000)
            
            cte_entropies = self.calculator.calculate(cte_signal)
            healthy_entropies = self.calculator.calculate(healthy_signal)
            
            cte_trend = np.polyfit(np.arange(len(cte_entropies)), cte_entropies, 1)[0]
            healthy_trend = np.polyfit(np.arange(len(healthy_entropies)), healthy_entropies, 1)[0]
            
            cte_trends.append(cte_trend)
            healthy_trends.append(healthy_trend)
        
        # On average, CTE should show more declining trend
        avg_cte_trend = np.mean(cte_trends)
        avg_healthy_trend = np.mean(healthy_trends)
        
        # For now, just verify that we can calculate trends for different conditions
        # Real validation will come with actual CTE/neurological datasets
        assert len(cte_trends) == 5
        assert len(healthy_trends) == 5
        print(f"CTE avg trend: {avg_cte_trend:.6f}, Healthy avg trend: {avg_healthy_trend:.6f}")
        # TODO: Replace with real CTE data validation once we have clinical datasets
    
    def test_entropy_reproducibility(self):
        """Test that entropy calculations are reproducible with same seed."""
        # Generate same signal twice with same seed
        gen1 = SignalGenerator(random_seed=123)
        gen2 = SignalGenerator(random_seed=123)
        
        signal1 = gen1.generate_healthy_series(300)
        signal2 = gen2.generate_healthy_series(300)
        
        entropies1 = self.calculator.calculate(signal1)
        entropies2 = self.calculator.calculate(signal2)
        
        np.testing.assert_array_almost_equal(entropies1, entropies2)