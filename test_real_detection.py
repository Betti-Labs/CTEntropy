#!/usr/bin/env python3
"""
Simple test to verify the clinical system can detect real differences
Tests on known different signal types to validate detection capability
"""

import numpy as np
import matplotlib.pyplot as plt
from clinical_ctentropy_system import ClinicalCTEntropySystem

def generate_test_signals():
    """Generate different types of test signals with known characteristics"""
    
    # Signal 1: Normal healthy-like signal (mixed frequencies)
    t = np.linspace(0, 10, 2560)  # 10 seconds at 256 Hz
    healthy_signal = (
        np.sin(2 * np.pi * 10 * t) +  # Alpha-like (10 Hz)
        0.5 * np.sin(2 * np.pi * 20 * t) +  # Beta-like (20 Hz)
        0.3 * np.random.randn(len(t))  # Some noise
    )
    
    # Signal 2: Epilepsy-like signal (high amplitude spikes + low entropy)
    epilepsy_signal = (
        0.5 * np.sin(2 * np.pi * 8 * t) +  # Lower frequency base
        3 * np.sin(2 * np.pi * 4 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0.8) +  # Spike-like activity
        0.1 * np.random.randn(len(t))  # Less noise (more regular)
    )
    
    # Signal 3: Alcoholism-like signal (altered frequency patterns)
    alcoholism_signal = (
        0.3 * np.sin(2 * np.pi * 6 * t) +  # Slower theta-like
        0.8 * np.sin(2 * np.pi * 15 * t) +  # Different beta pattern
        0.5 * np.random.randn(len(t))  # More noise
    )
    
    return {
        'healthy': healthy_signal,
        'epilepsy': epilepsy_signal, 
        'alcoholism': alcoholism_signal
    }

def test_clinical_detection():
    """Test if the clinical system can distinguish between different signal types"""
    
    print("🧪 Testing Clinical System Detection Capability")
    print("=" * 60)
    
    # Initialize clinical system
    clinical_system = ClinicalCTEntropySystem(
        facility_name="Detection Test Lab",
        physician_name="Dr. Test"
    )
    
    # Generate test signals
    signals = generate_test_signals()
    results = {}
    
    print("\n🔬 Analyzing different signal types...")
    
    for signal_type, signal in signals.items():
        print(f"\n--- Testing {signal_type.upper()} signal ---")
        
        # Analyze with clinical system
        result = clinical_system.analyze_patient_eeg(
            eeg_signal=signal,
            sampling_rate=256.0,
            patient_id=f"TEST_{signal_type.upper()}_001",
            user_id="TESTER"
        )
        
        if result['status'] == 'success':
            diagnosis = result['diagnosis']
            print(f"✅ Detected: {diagnosis['condition']}")
            print(f"   Confidence: {diagnosis['confidence']:.1f}%")
            
            results[signal_type] = {
                'detected_condition': diagnosis['condition'],
                'confidence': diagnosis['confidence'],
                'expected': signal_type.title()
            }
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            results[signal_type] = {'error': result.get('error', 'Unknown error')}
    
    # Analyze results
    print("\n" + "=" * 60)
    print("📊 DETECTION ANALYSIS")
    print("=" * 60)
    
    correct_detections = 0
    total_tests = 0
    
    for signal_type, result in results.items():
        if 'error' not in result:
            total_tests += 1
            expected = result['expected']
            detected = result['detected_condition']
            confidence = result['confidence']
            
            is_correct = (expected.lower() == detected.lower())
            if is_correct:
                correct_detections += 1
                status = "✅ CORRECT"
            else:
                status = "❌ INCORRECT"
            
            print(f"{signal_type.upper():12} | Expected: {expected:10} | Detected: {detected:10} | {confidence:5.1f}% | {status}")
    
    if total_tests > 0:
        accuracy = (correct_detections / total_tests) * 100
        print(f"\n🎯 Overall Detection Accuracy: {accuracy:.1f}% ({correct_detections}/{total_tests})")
        
        if accuracy >= 66:
            print("✅ GOOD: System shows real detection capability!")
        elif accuracy >= 33:
            print("⚠️  MIXED: System shows some detection capability but needs improvement")
        else:
            print("❌ POOR: System may not be detecting real differences")
    else:
        print("❌ No successful tests completed")
    
    # Test entropy differences
    print("\n" + "=" * 60)
    print("🧠 ENTROPY ANALYSIS")
    print("=" * 60)
    
    from ctentropy_platform.core.entropy import SymbolicEntropyCalculator
    calculator = SymbolicEntropyCalculator(window_size=25)
    
    for signal_type, signal in signals.items():
        entropies = calculator.calculate(signal)
        mean_entropy = np.mean(entropies)
        std_entropy = np.std(entropies)
        
        print(f"{signal_type.upper():12} | Mean Entropy: {mean_entropy:.3f} | Std: {std_entropy:.3f}")
    
    # Check if entropies are actually different
    entropy_values = []
    for signal_type, signal in signals.items():
        entropies = calculator.calculate(signal)
        entropy_values.append(np.mean(entropies))
    
    entropy_range = max(entropy_values) - min(entropy_values)
    print(f"\n📈 Entropy Range: {entropy_range:.3f}")
    
    if entropy_range > 0.2:
        print("✅ GOOD: Significant entropy differences detected between signal types")
    elif entropy_range > 0.1:
        print("⚠️  MODERATE: Some entropy differences detected")
    else:
        print("❌ POOR: Very small entropy differences - may not be meaningful")
    
    return results

if __name__ == "__main__":
    results = test_clinical_detection()