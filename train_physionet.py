"""
Train CTEntropy on Real PhysioNet EEG Data
Validate entropy signatures with clinical EEG recordings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from ctentropy_platform.data.physionet_loader import PhysioNetEEGLoader
from ctentropy_platform.core.entropy import SymbolicEntropyCalculator
from scipy.stats import entropy as scipy_entropy
from scipy.fftpack import fft

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_physionet_data():
    """Analyze PhysioNet EEG data with CTEntropy"""
    
    # Initialize components
    loader = PhysioNetEEGLoader()
    symbolic_calc = SymbolicEntropyCalculator()
    
    def calculate_spectral_entropy(signal, sampling_rate):
        """Simple spectral entropy calculation"""
        # Get power spectrum
        spectrum = np.abs(fft(signal))[:len(signal)//2]
        # Normalize to probability distribution
        spectrum = spectrum / np.sum(spectrum)
        # Calculate Shannon entropy
        return scipy_entropy(spectrum + 1e-9, base=2)
    
    print("🧠 CTEntropy Analysis on Real PhysioNet EEG Data")
    print("=" * 60)
    
    # Load data from available subjects
    print(f"Available subjects: {loader.subjects}")
    
    # Load data (limit to avoid memory issues)
    df = loader.load_all_subjects(max_subjects=5, max_files_per_subject=3)
    print(f"\nLoaded {len(df)} EEG recordings from {df['subject'].nunique()} subjects")
    
    # Analyze each recording
    results = []
    
    for _, row in df.iterrows():
        subject = row['subject']
        file = row['file']
        
        print(f"\nAnalyzing {subject} - {file}...")
        
        # Get signal
        signal = loader.get_signal(subject, file)
        if signal is None:
            continue
        
        # Calculate symbolic entropy (mean of all windows)
        symbolic_entropies = symbolic_calc.calculate(signal)
        symbolic_entropy = np.mean(symbolic_entropies)
        
        # Calculate spectral entropy  
        spectral_entropy = calculate_spectral_entropy(signal, row['sampling_rate'])
        
        # Store results
        result = {
            'subject': subject,
            'file': file,
            'run': file[-6:-4],  # Extract run number (R01, R02, etc.)
            'symbolic_entropy': symbolic_entropy,
            'spectral_entropy': spectral_entropy,
            'sampling_rate': row['sampling_rate'],
            'duration': row['duration'],
            'signal_length': len(signal)
        }
        
        results.append(result)
        
        print(f"  Symbolic Entropy: {symbolic_entropy:.3f}")
        print(f"  Spectral Entropy: {spectral_entropy:.3f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file = "results/physionet_entropy_analysis.csv"
    Path("results").mkdir(exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")
    
    # Statistical analysis
    print("\n📊 Statistical Summary:")
    print("=" * 40)
    
    # Overall statistics
    print("Overall Entropy Statistics:")
    print(f"Symbolic Entropy: {results_df['symbolic_entropy'].mean():.3f} ± {results_df['symbolic_entropy'].std():.3f}")
    print(f"Spectral Entropy: {results_df['spectral_entropy'].mean():.3f} ± {results_df['spectral_entropy'].std():.3f}")
    
    # Per-subject statistics
    print("\nPer-Subject Analysis:")
    subject_stats = results_df.groupby('subject').agg({
        'symbolic_entropy': ['mean', 'std', 'count'],
        'spectral_entropy': ['mean', 'std']
    }).round(3)
    
    print(subject_stats)
    
    # Visualization
    create_physionet_plots(results_df)
    
    return results_df

def create_physionet_plots(results_df):
    """Create visualization plots for PhysioNet analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CTEntropy Analysis on Real PhysioNet EEG Data', fontsize=16, fontweight='bold')
    
    # Plot 1: Symbolic Entropy by Subject
    ax1 = axes[0, 0]
    subjects = results_df['subject'].unique()
    symbolic_means = [results_df[results_df['subject'] == s]['symbolic_entropy'].mean() for s in subjects]
    symbolic_stds = [results_df[results_df['subject'] == s]['symbolic_entropy'].std() for s in subjects]
    
    bars1 = ax1.bar(subjects, symbolic_means, yerr=symbolic_stds, capsize=5, 
                    color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_title('Symbolic Entropy by Subject')
    ax1.set_ylabel('Symbolic Entropy')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Spectral Entropy by Subject
    ax2 = axes[0, 1]
    spectral_means = [results_df[results_df['subject'] == s]['spectral_entropy'].mean() for s in subjects]
    spectral_stds = [results_df[results_df['subject'] == s]['spectral_entropy'].std() for s in subjects]
    
    bars2 = ax2.bar(subjects, spectral_means, yerr=spectral_stds, capsize=5,
                    color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax2.set_title('Spectral Entropy by Subject')
    ax2.set_ylabel('Spectral Entropy')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Entropy Distribution
    ax3 = axes[1, 0]
    ax3.hist(results_df['symbolic_entropy'], bins=15, alpha=0.7, color='skyblue', 
             label='Symbolic', density=True)
    ax3.hist(results_df['spectral_entropy'], bins=15, alpha=0.7, color='lightcoral',
             label='Spectral', density=True)
    ax3.set_title('Entropy Distribution')
    ax3.set_xlabel('Entropy Value')
    ax3.set_ylabel('Density')
    ax3.legend()
    
    # Plot 4: Correlation Plot
    ax4 = axes[1, 1]
    ax4.scatter(results_df['symbolic_entropy'], results_df['spectral_entropy'], 
                alpha=0.6, color='purple')
    ax4.set_title('Symbolic vs Spectral Entropy')
    ax4.set_xlabel('Symbolic Entropy')
    ax4.set_ylabel('Spectral Entropy')
    
    # Add correlation coefficient
    corr = results_df['symbolic_entropy'].corr(results_df['spectral_entropy'])
    ax4.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = "results/physionet_entropy_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📈 Plots saved to {plot_file}")
    
    plt.show()

if __name__ == "__main__":
    results = analyze_physionet_data()