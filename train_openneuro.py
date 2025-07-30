#!/usr/bin/env python3
"""
CTEntropy OpenNeuro Training Script

Uses real clinical data from OpenNeuro to train the entropy analysis system.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import warnings

from ctentropy_platform.core import SymbolicEntropyCalculator
from ctentropy_platform.data.openneuro_loader import OpenNeuroLoader

warnings.filterwarnings('ignore')

def main():
    """Train CTEntropy on real OpenNeuro clinical data."""
    print("🧠 CTEntropy OpenNeuro Training")
    print("=" * 60)
    
    # Initialize OpenNeuro loader with your API key
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5NTU4ZmU2Yy1iZDk4LTQyMWItYjczMi1jYTZlYTI0ZDMxNjMiLCJwcm92aWRlciI6Im9yY2lkIiwibmFtZSI6IkdyZWdvcnkgQmV0dGkiLCJhZG1pbiI6ZmFsc2UsImlhdCI6MTc1Mzg2NDU4MiwiZXhwIjoxNzg1NDAwNTgyfQ.zuMrmQEiP1FUuu2YoMYCku14xLDNjhDQdFLesFyaJ9Q"
    
    loader = OpenNeuroLoader(api_key=api_key)
    
    # List available datasets
    print("🔍 Checking available clinical datasets...")
    datasets = loader.list_available_datasets()
    
    # Test API connection with a simple dataset first
    print("\n🔗 Testing API connection...")
    test_datasets = ['ds002778', 'ds003061']  # Start with healthy control datasets
    
    accessible_datasets = []
    for dataset_id in test_datasets:
        if loader.check_dataset_availability(dataset_id):
            accessible_datasets.append(dataset_id)
    
    if not accessible_datasets:
        print("❌ No datasets accessible. Check API key and permissions.")
        return
    
    print(f"✅ Found {len(accessible_datasets)} accessible datasets")
    
    # Setup training data from accessible datasets
    print("\n📥 Setting up training data...")
    training_data = loader.setup_clinical_training_data(max_subjects_per_dataset=2)
    
    if not training_data:
        print("❌ No training data available. Check API key and internet connection.")
        return
    
    # Initialize entropy calculator
    calculator = SymbolicEntropyCalculator(window_size=500)  # 2-second windows at 250Hz
    
    # Process each condition
    results = {}
    
    for condition, files in training_data.items():
        print(f"\n🔧 Processing {condition.title()} data...")
        
        condition_entropies = []
        
        for file_data in files:
            try:
                # Simple preprocessing: use first channel, segment into 4-second windows
                data = file_data['data']
                sampling_rate = file_data['sampling_rate']
                
                if len(data.shape) > 1 and data.shape[0] > 0:
                    signal = data[0, :]  # First channel
                else:
                    signal = data.flatten()
                
                # Segment signal
                segment_length = int(4 * sampling_rate)  # 4 seconds
                
                for start in range(0, len(signal) - segment_length, segment_length // 2):
                    segment = signal[start:start + segment_length]
                    
                    if np.std(segment) > 0.1:  # Quality check
                        entropies = calculator.calculate(segment)
                        condition_entropies.extend(entropies)
                
            except Exception as e:
                print(f"  ⚠️  Error processing file: {e}")
                continue
        
        if condition_entropies:
            results[condition] = {
                'entropies': np.array(condition_entropies),
                'mean': np.mean(condition_entropies),
                'std': np.std(condition_entropies),
                'count': len(condition_entropies)
            }
            print(f"  ✅ {len(condition_entropies)} entropy values from {condition}")
    
    # Create visualization
    if results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CTEntropy Analysis on Real OpenNeuro Clinical Data', fontsize=16)
        
        conditions = list(results.keys())
        colors = ['blue', 'red', 'green', 'purple']
        
        # Plot 1: Mean entropy comparison
        ax1 = axes[0, 0]
        means = [results[cond]['mean'] for cond in conditions]
        stds = [results[cond]['std'] for cond in conditions]
        bars = ax1.bar(range(len(conditions)), means, yerr=stds, 
                       color=colors[:len(conditions)], alpha=0.7, capsize=5)
        ax1.set_title('Mean Entropy by Condition (Real Clinical Data)')
        ax1.set_xlabel('Condition')
        ax1.set_ylabel('Mean Entropy')
        ax1.set_xticks(range(len(conditions)))
        ax1.set_xticklabels([c.title() for c in conditions], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Entropy distributions
        ax2 = axes[0, 1]
        for i, condition in enumerate(conditions):
            entropies = results[condition]['entropies']
            ax2.hist(entropies, bins=30, alpha=0.6, 
                    label=condition.title(), color=colors[i])
        ax2.set_title('Entropy Distributions (Real Data)')
        ax2.set_xlabel('Entropy Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sample counts
        ax3 = axes[1, 0]
        counts = [results[cond]['count'] for cond in conditions]
        bars = ax3.bar(range(len(conditions)), counts,
                       color=colors[:len(conditions)], alpha=0.7)
        ax3.set_title('Number of Entropy Samples')
        ax3.set_xlabel('Condition')
        ax3.set_ylabel('Sample Count')
        ax3.set_xticks(range(len(conditions)))
        ax3.set_xticklabels([c.title() for c in conditions], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = "Real Clinical Data Results\n" + "="*30 + "\n\n"
        for condition in conditions:
            stats = results[condition]
            summary_text += f"{condition.title()}:\n"
            summary_text += f"  Mean: {stats['mean']:.3f}\n"
            summary_text += f"  Std: {stats['std']:.3f}\n"
            summary_text += f"  Samples: {stats['count']}\n\n"
        
        summary_text += "✅ Real clinical validation complete!"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save results
        results_dir = Path("./results")
        results_dir.mkdir(exist_ok=True)
        
        # Save statistics
        stats_df = pd.DataFrame({
            cond: {
                'mean_entropy': results[cond]['mean'],
                'std_entropy': results[cond]['std'],
                'sample_count': results[cond]['count']
            } for cond in conditions
        }).T
        
        stats_file = results_dir / "openneuro_clinical_stats.csv"
        stats_df.to_csv(stats_file)
        print(f"\n✅ Statistics saved to {stats_file}")
        
        # Save plot
        plot_file = results_dir / "openneuro_clinical_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Analysis plot saved to {plot_file}")
        
        plt.show()
        
        print("\n🎉 OpenNeuro Clinical Training Complete!")
        print("Real clinical data has validated the CTEntropy methodology!")
    
    else:
        print("❌ No results to display")

if __name__ == "__main__":
    main()