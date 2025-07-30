#!/usr/bin/env python3
"""
CTEntropy Platform Demo

Demonstrates the core symbolic entropy calculation functionality
and shows the foundation for clinical neurological diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
from ctentropy_platform.core import SymbolicEntropyCalculator, SignalGenerator, ConditionType


def main():
    """Run the CTEntropy platform demonstration."""
    print("🧠 CTEntropy Diagnostic Platform - Core Engine Demo")
    print("=" * 60)
    
    # Initialize components
    calculator = SymbolicEntropyCalculator(window_size=50)
    generator = SignalGenerator(random_seed=42)
    
    print("\n1. Testing Core Symbolic Entropy Engine")
    print("-" * 40)
    
    # Generate test signals for all conditions
    conditions = [ConditionType.HEALTHY, ConditionType.CTE, 
                 ConditionType.ALZHEIMERS, ConditionType.DEPRESSION]
    
    results = {}
    
    for condition in conditions:
        print(f"Processing {condition.value.title()} signal...")
        
        # Generate signal
        signal = generator.generate_signal(condition, length=800)
        
        # Calculate entropy with metadata
        entropies, metadata = calculator.calculate_with_metadata(signal)
        
        # Detect entropy collapse
        collapse_info = calculator.detect_entropy_collapse(entropies)
        
        results[condition.value] = {
            'signal': signal,
            'entropies': entropies,
            'metadata': metadata,
            'collapse_info': collapse_info
        }
        
        print(f"  ✓ Mean entropy: {metadata['mean_entropy']:.3f}")
        print(f"  ✓ Entropy range: {metadata['entropy_range'][0]:.3f} - {metadata['entropy_range'][1]:.3f}")
        print(f"  ✓ Collapse detected: {collapse_info['collapse_detected']}")
        if collapse_info['collapse_detected']:
            print(f"    - Relative drop: {collapse_info['relative_drop']:.3f}")
            print(f"    - Trend slope: {collapse_info['slope']:.6f}")
    
    print("\n2. Entropy Pattern Analysis")
    print("-" * 40)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Original signals
    plt.subplot(2, 2, 1)
    for i, (condition_name, data) in enumerate(results.items()):
        plt.plot(data['signal'][:200], label=condition_name.title(), alpha=0.7)
    plt.title("Original Signals (First 200 samples)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Entropy time series
    plt.subplot(2, 2, 2)
    colors = ['blue', 'orange', 'green', 'purple']
    for i, (condition_name, data) in enumerate(results.items()):
        plt.plot(data['entropies'], label=condition_name.title(), 
                color=colors[i], linewidth=2)
    plt.title("Symbolic Entropy Patterns")
    plt.xlabel("Time Window")
    plt.ylabel("Entropy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Entropy statistics
    plt.subplot(2, 2, 3)
    condition_names = list(results.keys())
    mean_entropies = [results[name]['metadata']['mean_entropy'] for name in condition_names]
    std_entropies = [results[name]['metadata']['std_entropy'] for name in condition_names]
    
    x_pos = np.arange(len(condition_names))
    plt.bar(x_pos, mean_entropies, yerr=std_entropies, capsize=5, 
            color=colors, alpha=0.7)
    plt.title("Mean Entropy by Condition")
    plt.xlabel("Condition")
    plt.ylabel("Mean Entropy")
    plt.xticks(x_pos, [name.title() for name in condition_names], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Collapse detection summary
    plt.subplot(2, 2, 4)
    collapse_detected = [results[name]['collapse_info']['collapse_detected'] for name in condition_names]
    relative_drops = [results[name]['collapse_info']['relative_drop'] for name in condition_names]
    
    bars = plt.bar(x_pos, relative_drops, color=colors, alpha=0.7)
    plt.title("Entropy Collapse Detection")
    plt.xlabel("Condition")
    plt.ylabel("Relative Entropy Drop")
    plt.xticks(x_pos, [name.title() for name in condition_names], rotation=45)
    
    # Add collapse detection indicators
    for i, (bar, detected) in enumerate(zip(bars, collapse_detected)):
        if detected:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    '⚠️', ha='center', va='bottom', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print("\n3. Clinical Insights")
    print("-" * 40)
    
    # Analyze patterns
    for condition_name, data in results.items():
        collapse_info = data['collapse_info']
        metadata = data['metadata']
        
        print(f"\n{condition_name.title()} Analysis:")
        print(f"  • Entropy stability: {metadata['std_entropy']:.3f}")
        print(f"  • Trend slope: {collapse_info['slope']:.6f}")
        
        if collapse_info['collapse_detected']:
            print(f"  • ⚠️  ENTROPY COLLAPSE DETECTED")
            print(f"    - Initial entropy: {collapse_info['initial_entropy']:.3f}")
            print(f"    - Final entropy: {collapse_info['final_entropy']:.3f}")
            print(f"    - Relative drop: {collapse_info['relative_drop']:.1%}")
        else:
            print(f"  • ✓ Entropy patterns stable")
    
    print("\n4. Next Steps for Clinical Validation")
    print("-" * 40)
    print("🔬 CRITICAL: Replace synthetic signals with real clinical data:")
    print("  • EEG datasets from CTE patients (sports injuries, military)")
    print("  • Alzheimer's EEG data from clinical trials")
    print("  • Depression EEG from psychiatric studies")
    print("  • Healthy control groups matched by age/demographics")
    print("\n📊 Recommended datasets:")
    print("  • PhysioNet EEG databases")
    print("  • OpenNeuro neurological datasets") 
    print("  • ADNI (Alzheimer's Disease Neuroimaging Initiative)")
    print("  • TBI datasets from DoD/VA studies")
    
    print("\n✅ Task 1 Complete: Core entropy engine ready for real data!")
    print("🚀 Ready to move to Task 2: Enhanced entropy methods")
    
    plt.show()


if __name__ == "__main__":
    main()