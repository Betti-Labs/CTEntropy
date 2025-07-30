"""
Unit tests for signal generation utilities.
"""

import pytest
import numpy as np
from ctentropy_platform.core.signals import (
    SignalGenerator, ConditionType,
    generate_healthy_series, generate_cte_like_series,
    generate_alzheimers_series, generate_depression_series
)


class TestSignalGenerator:
    """Test suite for SignalGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SignalGenerator(random_seed=42)
    
    def test_initialization(self):
        """Test signal generator initialization."""
        # Test with seed
        gen_with_seed = SignalGenerator(random_seed=123)
        assert gen_with_seed is not None
        
        # Test without seed
        gen_without_seed = SignalGenerator()
        assert gen_without_seed is not None
    
    def test_generate_healthy_series(self):
        """Test healthy signal generation."""
        signal = self.generator.generate_healthy_series(length=500)
        
        # Check basic properties
        assert isinstance(signal, np.ndarray)
        assert len(signal) == 500
        assert signal.dtype == np.float64
        
        # Check signal characteristics
        assert np.std(signal) > 0  # Should have variability
        assert not np.all(signal == signal[0])  # Should not be constant
    
    def test_generate_cte_like_series(self):
        """Test CTE-like signal generation."""
        signal = self.generator.generate_cte_like_series(length=400)
        
        assert isinstance(signal, np.ndarray)
        assert len(signal) == 400
        
        # CTE signal should show decay pattern
        first_half_mean = np.mean(np.abs(signal[:200]))
        second_half_mean = np.mean(np.abs(signal[200:]))
        assert first_half_mean > second_half_mean  # Decay pattern
    
    def test_generate_alzheimers_series(self):
        """Test Alzheimer's-like signal generation."""
        signal = self.generator.generate_alzheimers_series(length=600)
        
        assert isinstance(signal, np.ndarray)
        assert len(signal) == 600
        
        # Should show gradual decline
        first_quarter = np.mean(np.abs(signal[:150]))
        last_quarter = np.mean(np.abs(signal[450:]))
        assert first_quarter > last_quarter  # Gradual decline
    
    def test_generate_depression_series(self):
        """Test depression-like signal generation."""
        signal = self.generator.generate_depression_series(length=300)
        
        assert isinstance(signal, np.ndarray)
        assert len(signal) == 300
        
        # Depression signal should have lower variability
        healthy_signal = self.generator.generate_healthy_series(length=300)
        assert np.std(signal) < np.std(healthy_signal)
    
    def test_generate_signal_with_enum(self):
        """Test signal generation using ConditionType enum."""
        for condition in ConditionType:
            signal = self.generator.generate_signal(condition, length=200)
            assert isinstance(signal, np.ndarray)
            assert len(signal) == 200
    
    def test_generate_signal_invalid_condition(self):
        """Test signal generation with invalid condition."""
        with pytest.raises(ValueError, match="Unknown condition type"):
            self.generator.generate_signal("invalid_condition", length=100)
    
    def test_generate_batch(self):
        """Test batch signal generation."""
        conditions = [ConditionType.HEALTHY, ConditionType.CTE]
        batch = self.generator.generate_batch(
            conditions, length=150, samples_per_condition=3
        )
        
        # Check batch structure
        assert isinstance(batch, dict)
        assert len(batch) == 2
        assert 'healthy' in batch
        assert 'cte' in batch
        
        # Check each condition has correct number of samples
        for condition_name, signals in batch.items():
            assert len(signals) == 3
            for signal in signals:
                assert isinstance(signal, np.ndarray)
                assert len(signal) == 150
    
    def test_generate_batch_with_string_conditions(self):
        """Test batch generation with string condition names."""
        conditions = ['healthy', 'depression']
        batch = self.generator.generate_batch(conditions, length=100)
        
        assert 'healthy' in batch
        assert 'depression' in batch
        assert len(batch['healthy']) == 1
        assert len(batch['depression']) == 1
    
    def test_reproducibility_with_seed(self):
        """Test that signals are reproducible with same seed."""
        gen1 = SignalGenerator(random_seed=999)
        gen2 = SignalGenerator(random_seed=999)
        
        signal1 = gen1.generate_healthy_series(200)
        signal2 = gen2.generate_healthy_series(200)
        
        np.testing.assert_array_almost_equal(signal1, signal2)
    
    def test_different_seeds_produce_different_signals(self):
        """Test that different seeds produce different signals."""
        gen1 = SignalGenerator(random_seed=111)
        gen2 = SignalGenerator(random_seed=222)
        
        signal1 = gen1.generate_healthy_series(200)
        signal2 = gen2.generate_healthy_series(200)
        
        # Signals should be different
        assert not np.allclose(signal1, signal2)


class TestLegacyFunctions:
    """Test suite for legacy signal generation functions."""
    
    def test_legacy_healthy_series(self):
        """Test legacy generate_healthy_series function."""
        signal = generate_healthy_series(length=300)
        
        assert isinstance(signal, np.ndarray)
        assert len(signal) == 300
        assert np.std(signal) > 0
    
    def test_legacy_cte_series(self):
        """Test legacy generate_cte_like_series function."""
        signal = generate_cte_like_series(length=250)
        
        assert isinstance(signal, np.ndarray)
        assert len(signal) == 250
        
        # Should show decay
        first_half = np.mean(np.abs(signal[:125]))
        second_half = np.mean(np.abs(signal[125:]))
        assert first_half > second_half
    
    def test_legacy_alzheimers_series(self):
        """Test legacy generate_alzheimers_series function."""
        signal = generate_alzheimers_series(length=400)
        
        assert isinstance(signal, np.ndarray)
        assert len(signal) == 400
    
    def test_legacy_depression_series(self):
        """Test legacy generate_depression_series function."""
        signal = generate_depression_series(length=350)
        
        assert isinstance(signal, np.ndarray)
        assert len(signal) == 350


class TestSignalCharacteristics:
    """Test suite for verifying signal characteristics match research expectations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SignalGenerator(random_seed=42)
    
    def test_signal_frequency_content(self):
        """Test that signals have expected frequency characteristics."""
        # Generate signals
        healthy = self.generator.generate_healthy_series(1000)
        cte = self.generator.generate_cte_like_series(1000)
        alzheimers = self.generator.generate_alzheimers_series(1000)
        depression = self.generator.generate_depression_series(1000)
        
        # All signals should have non-zero frequency content
        for signal in [healthy, cte, alzheimers, depression]:
            fft_signal = np.abs(np.fft.fft(signal))
            assert np.sum(fft_signal) > 0
    
    def test_signal_amplitude_ranges(self):
        """Test that signals have reasonable amplitude ranges."""
        conditions = [ConditionType.HEALTHY, ConditionType.CTE, 
                     ConditionType.ALZHEIMERS, ConditionType.DEPRESSION]
        
        for condition in conditions:
            signal = self.generator.generate_signal(condition, length=500)
            
            # Signals should have reasonable amplitude ranges
            assert np.min(signal) > -10  # Not too negative
            assert np.max(signal) < 10   # Not too positive
            assert np.std(signal) > 0.01  # Some variability
    
    def test_condition_specific_patterns(self):
        """Test that each condition has its expected pattern characteristics."""
        # Generate multiple samples for statistical testing
        n_samples = 5
        length = 500
        
        # Test CTE decay pattern
        cte_signals = [self.generator.generate_cte_like_series(length) 
                      for _ in range(n_samples)]
        
        for signal in cte_signals:
            # CTE should show exponential-like decay
            first_quarter = np.mean(np.abs(signal[:length//4]))
            last_quarter = np.mean(np.abs(signal[-length//4:]))
            assert first_quarter > last_quarter * 1.2  # Significant decay
        
        # Test Alzheimer's linear decay
        alz_signals = [self.generator.generate_alzheimers_series(length) 
                      for _ in range(n_samples)]
        
        for signal in alz_signals:
            # Should show more gradual, linear-like decline
            first_half = np.mean(np.abs(signal[:length//2]))
            second_half = np.mean(np.abs(signal[length//2:]))
            assert first_half > second_half  # Gradual decline
        
        # Test depression low variability
        dep_signals = [self.generator.generate_depression_series(length) 
                      for _ in range(n_samples)]
        healthy_signals = [self.generator.generate_healthy_series(length) 
                          for _ in range(n_samples)]
        
        dep_std = np.mean([np.std(signal) for signal in dep_signals])
        healthy_std = np.mean([np.std(signal) for signal in healthy_signals])
        
        assert dep_std < healthy_std  # Depression should have lower variability