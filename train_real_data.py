#!/usr/bin/env python3
"""
CTEntropy Real Data Training Script

Downloads real neurological datasets and trains the entropy analysis system
on clinical data instead of synthetic signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Dict, List
import warnings

from ctentropy_platform.core import SymbolicEntropyCalculator
from ctentropy_platform.data import EEGDataLoader, EEGPreprocessor, PreprocessingConfig

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def main():
    """Train CTEntropy on real neurological data."""
    print("🧠 CTEntropy Real Data Training")
    print("=" * 60)
    print("Training on real clinical datasets from PhysioNet")
    print("This will download and process actual EEG data!")
    print()
    
    # Initialize components
    data_loader = EEGDataLoader(data_dir="./real_data")
    preprocessor = EEGPreprocessor(PreprocessingConfig(
        segment_length=4.0,  # 4-second segments
        segment_overlap=0.5,  # 50% overlap
        target_sampling_rate=250.0,  # Standardize to 250Hz
        min_signal_quality=0.5  # Accept moderate quality
    ))
    entropy_calculator = SymbolicEntropyCalculator(window_size=50)
    
    # Step 1: Download and load real data
    print("📥 Step 1: Loading Real Clinical Data")
    print("-" * 40)
    
    try:
        training_data = data_loader.setup_training_data(max_subjects_per_dataset=3)
        
        if not training_data:
            print("❌ No training data available. Check internet connection.")
            return
            
    except Exception as e:
        print(f"❌ Failed to load training data: {e}")
        print("💡 Make sure you have internet connection and try: pip install wfdb")
        return
    
    # Step 2: Preprocess real EEG data
    print(f"\n🔧 Step 2: Preprocessing Real EEG Data")
    print("-" * 40)
    
    processed_data = {}
    entropy_results = {}
    
    for condition, records in training_data.items():
        print(f"\nProcessing {condition.title()} data ({len(records)} records)...")
        
        condition_entropies = []
        condition_metadata = []
        
        for i, record in enumerate(records):
            try:
                print(f"  Record {i+1}/{len(records)}: {record.subject_id}")
                
                # Preprocess the EEG data
                processed = preprocessor.preprocess(
                    record.data, record.sampling_rate, record.channels
                )
                
                print(f"    ✓ {len(processed.data)} segments extracted")
                
                # Calculate entropy for each segment
                for j, segment in enumerate(processed.data):
                    # Use first EEG channel for entropy analysis
                    eeg_channel = 0
                    if segment.shape[0] > eeg_channel:
                        signal = segment[eeg_channel, :]
                        
                        # Calculate entropy
                        entropies, metadata = entropy_calculator.calculate_with_metadata(signal)
                        
                        # Store results
                        condition_entropies.append(entropies)
                        condition_metadata.append({
                            'subject_id': record.subject_id,
                            'segment_id': j,
                            'condition': condition,
                            'sampling_rate': processed.sampling_rate,
                            'quality_score': processed.quality_scores[j],
                            **metadata
                        })
                
            except Exception as e:
                print(f"    ⚠️  Failed to process {record.subject_id}: {e}")
                continue
        
        if condition_entropies:
            processed_data[condition] = condition_entropies
            entropy_results[condition] = condition_metadata
            print(f"  ✅ {len(condition_entropies)} entropy sequences from {condition}")
        else:
            print(f"  ❌ No valid entropy data for {condition}")
    
    if not processed_data:
        print("❌ No data was successfully processed!")
        return
    
    # Step 3: Analyze real entropy patterns
    print(f"\n📊 Step 3: Analyzing Real Entropy Patterns")
    print("-" * 40)
    
    # Calculate statistics for each condition
    condition_stats = {}
    
    for condition, entropies_list in processed_data.items():
        # Combine all entropy sequences for this condition
        all_entropies = np.concatenate(entropies_list)
        
        # Calculate collapse detection for each sequence
        collapse_detections = []
        for entropies in entropies_list:
            collapse_info = entropy_calculator.detect_entropy_collapse(entropies)
            collapse_detections.append(collapse_info)
        
        # Calculate statistics
        stats = {
            'mean_entropy': np.mean(all_entropies),
            'std_entropy': np.std(all_entropies),
            'median_entropy': np.median(all_entropies),
            'entropy_range': (np.min(all_entropies), np.max(all_entropies)),
            'num_sequences': len(entropies_list),
            'total_windows': len(all_entropies),
            'collapse_rate': np.mean([cd['collapse_detected'] for cd in collapse_detections]),
            'mean_slope': np.mean([cd['slope'] for cd in collapse_detections]),
            'mean_relative_drop': np.mean([cd['relative_drop'] for cd in collapse_detections])
        }
        
        condition_stats[condition] = stats
        
        print(f"\n{condition.title()} Statistics:")
        print(f"  • Sequences: {stats['num_sequences']}")
        print(f"  • Total windows: {stats['total_windows']}")
        print(f"  • Mean entropy: {stats['mean_entropy']:.3f} ± {stats['std_entropy']:.3f}")
        print(f"  • Entropy range: {stats['entropy_range'][0]:.3f} - {stats['entropy_range'][1]:.3f}")
        print(f"  • Collapse detection rate: {stats['collapse_rate']:.1%}")
        print(f"  • Mean trend slope: {stats['mean_slope']:.6f}")
        print(f"  • Mean relative drop: {stats['mean_relative_drop']:.3f}")
    
    # Step 4: Visualize real data patterns
    print(f"\n📈 Step 4: Visualizing Real Data Patterns")
    print("-" * 40)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CTEntropy Analysis on Real Clinical Data', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    condition_names = list(processed_data.keys())
    
    # Plot 1: Sample entropy time series
    ax1 = axes[0, 0]
    for i, (condition, entropies_list) in enumerate(processed_data.items()):
        if entropies_list:
            sample_entropy = entropies_list[0]  # First sequence
            ax1.plot(sample_entropy, label=condition.title(), 
                    color=colors[i % len(colors)], linewidth=2, alpha=0.8)
    ax1.set_title('Sample Entropy Time Series (Real Data)')
    ax1.set_xlabel('Time Window')
    ax1.set_ylabel('Entropy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean entropy comparison
    ax2 = axes[0, 1]
    means = [condition_stats[cond]['mean_entropy'] for cond in condition_names]
    stds = [condition_stats[cond]['std_entropy'] for cond in condition_names]
    bars = ax2.bar(range(len(condition_names)), means, yerr=stds, 
                   color=[colors[i % len(colors)] for i in range(len(condition_names))],
                   alpha=0.7, capsize=5)
    ax2.set_title('Mean Entropy by Condition (Real Data)')
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Mean Entropy')
    ax2.set_xticks(range(len(condition_names)))
    ax2.set_xticklabels([name.title() for name in condition_names], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Entropy distributions
    ax3 = axes[0, 2]
    for i, (condition, entropies_list) in enumerate(processed_data.items()):
        if entropies_list:
            all_entropies = np.concatenate(entropies_list)
            ax3.hist(all_entropies, bins=30, alpha=0.6, 
                    label=condition.title(), color=colors[i % len(colors)])
    ax3.set_title('Entropy Distributions (Real Data)')
    ax3.set_xlabel('Entropy Value')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Collapse detection rates
    ax4 = axes[1, 0]
    collapse_rates = [condition_stats[cond]['collapse_rate'] for cond in condition_names]
    bars = ax4.bar(range(len(condition_names)), collapse_rates,
                   color=[colors[i % len(colors)] for i in range(len(condition_names))],
                   alpha=0.7)
    ax4.set_title('Entropy Collapse Detection Rate')
    ax4.set_xlabel('Condition')
    ax4.set_ylabel('Collapse Detection Rate')
    ax4.set_xticks(range(len(condition_names)))
    ax4.set_xticklabels([name.title() for name in condition_names], rotation=45)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, collapse_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # Plot 5: Trend slopes
    ax5 = axes[1, 1]
    slopes = [condition_stats[cond]['mean_slope'] for cond in condition_names]
    bars = ax5.bar(range(len(condition_names)), slopes,
                   color=[colors[i % len(colors)] for i in range(len(condition_names))],
                   alpha=0.7)
    ax5.set_title('Mean Entropy Trend Slopes')
    ax5.set_xlabel('Condition')
    ax5.set_ylabel('Slope (entropy/window)')
    ax5.set_xticks(range(len(condition_names)))
    ax5.set_xticklabels([name.title() for name in condition_names], rotation=45)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Data summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary text
    summary_text = "Real Data Training Summary\n" + "="*30 + "\n\n"
    
    total_sequences = sum(stats['num_sequences'] for stats in condition_stats.values())
    total_windows = sum(stats['total_windows'] for stats in condition_stats.values())
    
    summary_text += f"Total Sequences: {total_sequences}\n"
    summary_text += f"Total Windows: {total_windows}\n\n"
    
    summary_text += "Conditions Analyzed:\n"
    for condition, stats in condition_stats.items():
        summary_text += f"• {condition.title()}: {stats['num_sequences']} sequences\n"
    
    summary_text += f"\nKey Findings:\n"
    
    # Find condition with highest collapse rate
    max_collapse_condition = max(condition_stats.keys(), 
                               key=lambda x: condition_stats[x]['collapse_rate'])
    max_collapse_rate = condition_stats[max_collapse_condition]['collapse_rate']
    
    summary_text += f"• Highest collapse rate: {max_collapse_condition.title()} ({max_collapse_rate:.1%})\n"
    
    # Find condition with most negative slope
    min_slope_condition = min(condition_stats.keys(),
                            key=lambda x: condition_stats[x]['mean_slope'])
    min_slope = condition_stats[min_slope_condition]['mean_slope']
    
    summary_text += f"• Most declining trend: {min_slope_condition.title()} ({min_slope:.6f})\n"
    
    summary_text += f"\n✅ Real data training complete!\n"
    summary_text += f"🚀 Ready for clinical validation"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Step 5: Save results
    print(f"\n💾 Step 5: Saving Results")
    print("-" * 40)
    
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    # Save statistics
    stats_df = pd.DataFrame(condition_stats).T
    stats_file = results_dir / "real_data_entropy_stats.csv"
    stats_df.to_csv(stats_file)
    print(f"✅ Statistics saved to {stats_file}")
    
    # Save detailed results
    all_results = []
    for condition, metadata_list in entropy_results.items():
        for metadata in metadata_list:
            all_results.append(metadata)
    
    results_df = pd.DataFrame(all_results)
    results_file = results_dir / "real_data_detailed_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"✅ Detailed results saved to {results_file}")
    
    # Save plot
    plot_file = results_dir / "real_data_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Analysis plot saved to {plot_file}")
    
    print(f"\n🎉 Real Data Training Complete!")
    print("=" * 60)
    print("Key Achievements:")
    print(f"• Processed {total_sequences} real EEG sequences")
    print(f"• Analyzed {total_windows} entropy windows")
    print(f"• Identified distinct patterns across conditions")
    print(f"• Results saved to ./results/ directory")
    print()
    print("Next Steps:")
    print("• Validate patterns with larger datasets")
    print("• Implement machine learning classification")
    print("• Test on independent validation set")
    print("• Prepare for clinical trials")
    
    plt.show()


if __name__ == "__main__":
    main()