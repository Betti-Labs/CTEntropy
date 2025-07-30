"""
CTEntropy Analysis on REAL UCI Alcoholism EEG Data
Test entropy signatures on actual alcoholic vs control EEG recordings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy import stats
import seaborn as sns
from pathlib import Path
import logging

from ctentropy_platform.data.uci_alcoholism_loader import UCIAlcoholismLoader
from ctentropy_platform.core.entropy import SymbolicEntropyCalculator
from scipy.stats import entropy as scipy_entropy
from scipy.fftpack import fft

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealAlcoholismCTEntropyAnalyzer:
    """Analyze REAL alcoholism EEG data with CTEntropy"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = []
        
    def extract_alcoholism_features(self, signal, sampling_rate):
        """Extract entropy features optimized for alcoholism detection"""
        
        # Initialize entropy calculator
        symbolic_calc = SymbolicEntropyCalculator(window_size=25)  # Shorter for 1-second signals
        
        # Multi-scale entropy analysis
        entropies_10 = SymbolicEntropyCalculator(window_size=10).calculate(signal)
        entropies_25 = symbolic_calc.calculate(signal)
        entropies_50 = SymbolicEntropyCalculator(window_size=50).calculate(signal)
        
        # Spectral entropy
        spectrum = np.abs(fft(signal))[:len(signal)//2]
        spectrum = spectrum / np.sum(spectrum)
        spectral_entropy = scipy_entropy(spectrum + 1e-9, base=2)
        
        # Alcoholism-specific features
        # Neural flexibility (key difference in addiction)
        neural_flexibility = np.std(entropies_25) / (np.mean(entropies_25) + 1e-9)
        
        # Frequency band analysis (alcoholism affects specific bands)
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)[:len(signal)//2]
        
        # Alpha band power (8-12 Hz) - reduced in alcoholics
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        alpha_power = np.sum(spectrum[alpha_mask])
        
        # Beta band power (13-30 Hz) - increased in alcoholics
        beta_mask = (freqs >= 13) & (freqs <= 30)
        beta_power = np.sum(spectrum[beta_mask])
        
        # Theta band power (4-8 Hz) - altered in alcoholics
        theta_mask = (freqs >= 4) & (freqs <= 8)
        theta_power = np.sum(spectrum[theta_mask])
        
        # Alpha/Beta ratio (diagnostic marker)
        alpha_beta_ratio = alpha_power / (beta_power + 1e-9)
        
        # Signal characteristics
        signal_amplitude = np.std(signal)
        signal_peak_ratio = np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-9)
        
        features = {
            # Multi-scale entropy
            'entropy_10_mean': np.mean(entropies_10),
            'entropy_25_mean': np.mean(entropies_25),
            'entropy_50_mean': np.mean(entropies_50) if len(entropies_50) > 0 else np.mean(entropies_25),
            
            # Entropy variability (neural flexibility)
            'entropy_std': np.std(entropies_25),
            'neural_flexibility': neural_flexibility,
            
            # Spectral features
            'spectral_entropy': spectral_entropy,
            'alpha_power': alpha_power,
            'beta_power': beta_power,
            'theta_power': theta_power,
            'alpha_beta_ratio': alpha_beta_ratio,
            
            # Signal characteristics
            'signal_amplitude': signal_amplitude,
            'signal_peak_ratio': signal_peak_ratio
        }
        
        return features
    
    def analyze_real_alcoholism_data(self):
        """Analyze REAL UCI alcoholism patient data"""
        
        print("🍷 CTEntropy Analysis on REAL UCI Alcoholism Data")
        print("=" * 60)
        
        # Load real alcoholism data
        loader = UCIAlcoholismLoader()
        
        # Extract more subjects if needed
        if len(loader.subjects) < 10:
            print("Extracting more subjects for better analysis...")
            loader.extract_more_subjects(num_subjects=8)
        
        df = loader.load_all_subjects(max_subjects=10, max_files_per_subject=10)
        
        if len(df) == 0:
            print("❌ No alcoholism data found. Make sure UCI dataset is properly extracted!")
            return None
        
        print(f"Loaded {len(df)} REAL EEG recordings from {df['subject_id'].nunique()} subjects")
        print(f"Conditions: {df['condition'].value_counts().to_dict()}")
        
        # Extract features for each recording
        features_list = []
        
        for idx, row in df.iterrows():
            subject_id = row['subject_id']
            file = row['file']
            condition = row['condition']
            
            print(f"Analyzing {subject_id} - {file} ({condition})")
            
            # Get signal
            signal = loader.get_signal(subject_id, file)
            if signal is None:
                continue
            
            # Extract entropy features
            features = self.extract_alcoholism_features(signal, row['sampling_rate'])
            features['subject_id'] = subject_id
            features['file'] = file
            features['condition'] = condition
            features['is_alcoholic'] = row['is_alcoholic']
            
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Store feature names
        self.feature_names = [col for col in features_df.columns 
                            if col not in ['subject_id', 'file', 'condition', 'is_alcoholic']]
        
        # Save features
        features_file = "results/real_alcoholism_entropy_features.csv"
        Path("results").mkdir(exist_ok=True)
        features_df.to_csv(features_file, index=False)
        print(f"✅ Features saved to {features_file}")
        
        return features_df
    
    def train_real_alcoholism_detector(self, features_df):
        """Train alcoholism detection model on REAL data"""
        
        print("\n🤖 Training REAL Alcoholism Detection Model...")
        
        # Check data distribution
        condition_counts = features_df['condition'].value_counts()
        print(f"Data distribution: {condition_counts.to_dict()}")
        
        if len(condition_counts) < 2:
            print("⚠️  Only one condition found. Cannot train binary classifier.")
            return self.analyze_patterns_only(features_df)
        
        # Prepare features and labels
        X = features_df[self.feature_names]
        y = features_df['is_alcoholic']
        
        print(f"Features shape: {X.shape}")
        print(f"Label distribution: {y.value_counts().to_dict()}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Results
        accuracy = self.model.score(X_test, y_test)
        print(f"REAL Alcoholism Detection Accuracy: {accuracy:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Control', 'Alcoholic']))
        
        return {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_importance': self.model.feature_importances_
        }
    
    def analyze_patterns_only(self, features_df):
        """Analyze entropy patterns when we can't train classifier"""
        
        print("\n📊 Analyzing Entropy Patterns in REAL Alcoholism Data...")
        
        # Statistical analysis
        print("\nEntropy Statistics by Condition:")
        for condition in features_df['condition'].unique():
            subset = features_df[features_df['condition'] == condition]
            print(f"\n{condition} (n={len(subset)}):")
            
            for feature in ['entropy_25_mean', 'neural_flexibility', 'spectral_entropy', 'alpha_beta_ratio']:
                if feature in features_df.columns:
                    mean_val = subset[feature].mean()
                    std_val = subset[feature].std()
                    print(f"  {feature}: {mean_val:.3f} ± {std_val:.3f}")
        
        return {'pattern_analysis': True}
    
    def statistical_analysis(self, features_df):
        """Perform statistical analysis between groups"""
        
        print("\n📊 Statistical Analysis: Control vs Alcoholic (REAL DATA)")
        print("=" * 60)
        
        control_data = features_df[features_df['condition'] == 'Control']
        alcoholic_data = features_df[features_df['condition'] == 'Alcoholic']
        
        if len(control_data) == 0 or len(alcoholic_data) == 0:
            print("⚠️  Need both control and alcoholic data for comparison")
            return
        
        # Key entropy measures
        measures = ['entropy_25_mean', 'neural_flexibility', 'spectral_entropy', 'alpha_beta_ratio']
        
        for measure in measures:
            if measure in features_df.columns:
                control_vals = control_data[measure].values
                alcoholic_vals = alcoholic_data[measure].values
                
                # T-test
                ttest = stats.ttest_ind(control_vals, alcoholic_vals)
                
                # Effect size
                def cohens_d(x1, x2):
                    n1, n2 = len(x1), len(x2)
                    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
                    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
                    return (np.mean(x1) - np.mean(x2)) / s_pooled
                
                effect_size = cohens_d(control_vals, alcoholic_vals)
                
                print(f"\n{measure}:")
                print(f"  Control: {np.mean(control_vals):.3f} ± {np.std(control_vals):.3f}")
                print(f"  Alcoholic: {np.mean(alcoholic_vals):.3f} ± {np.std(alcoholic_vals):.3f}")
                print(f"  T-test: t={ttest.statistic:.3f}, p={ttest.pvalue:.6f}")
                print(f"  Effect size: {effect_size:.3f}")
                print(f"  Significant: {'YES' if ttest.pvalue < 0.05 else 'NO'}")
    
    def create_real_alcoholism_visualization(self, features_df, results=None):
        """Create visualizations for REAL alcoholism analysis"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CTEntropy Analysis on REAL UCI Alcoholism Data', fontsize=16, fontweight='bold')
        
        # Check if we have both conditions
        conditions = features_df['condition'].unique()
        
        if len(conditions) >= 2:
            # 1. Entropy comparison
            ax1 = axes[0, 0]
            control_entropy = features_df[features_df['condition'] == 'Control']['entropy_25_mean']
            alcoholic_entropy = features_df[features_df['condition'] == 'Alcoholic']['entropy_25_mean']
            
            data_to_plot = [control_entropy, alcoholic_entropy]
            box_plot = ax1.boxplot(data_to_plot, labels=['Control', 'Alcoholic'], patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            box_plot['boxes'][1].set_facecolor('lightcoral')
            
            ax1.set_title('Symbolic Entropy: Control vs Alcoholic')
            ax1.set_ylabel('Symbolic Entropy')
        else:
            ax1 = axes[0, 0]
            ax1.text(0.5, 0.5, f'Only {conditions[0]} data\navailable', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Data Limitation')
        
        # 2. Subject comparison
        ax2 = axes[0, 1]
        subjects = features_df['subject_id'].unique()
        subject_entropies = [features_df[features_df['subject_id'] == s]['entropy_25_mean'].mean() 
                           for s in subjects]
        
        colors = ['red' if 'co2a' in s or 'co3a' in s else 'blue' for s in subjects]
        ax2.bar(range(len(subjects)), subject_entropies, color=colors, alpha=0.7)
        ax2.set_title('Mean Entropy by Subject')
        ax2.set_ylabel('Entropy')
        ax2.set_xticks(range(len(subjects)))
        ax2.set_xticklabels([s[-3:] for s in subjects], rotation=45)
        
        # 3. Alpha/Beta ratio
        ax3 = axes[0, 2]
        if len(conditions) >= 2:
            control_ratio = features_df[features_df['condition'] == 'Control']['alpha_beta_ratio']
            alcoholic_ratio = features_df[features_df['condition'] == 'Alcoholic']['alpha_beta_ratio']
            
            ax3.hist(control_ratio, alpha=0.7, label='Control', bins=10, color='blue')
            ax3.hist(alcoholic_ratio, alpha=0.7, label='Alcoholic', bins=10, color='red')
            ax3.set_title('Alpha/Beta Ratio Distribution')
            ax3.set_xlabel('Alpha/Beta Ratio')
            ax3.legend()
        else:
            ax3.hist(features_df['alpha_beta_ratio'], bins=15, alpha=0.7)
            ax3.set_title('Alpha/Beta Ratio Distribution')
        
        # 4. Feature space
        ax4 = axes[1, 0]
        for condition in conditions:
            subset = features_df[features_df['condition'] == condition]
            ax4.scatter(subset['entropy_25_mean'], subset['spectral_entropy'], 
                       label=condition, alpha=0.7)
        ax4.set_xlabel('Symbolic Entropy')
        ax4.set_ylabel('Spectral Entropy')
        ax4.set_title('REAL Alcoholism Feature Space')
        ax4.legend()
        
        # 5. Neural flexibility
        ax5 = axes[1, 1]
        if len(conditions) >= 2:
            control_flex = features_df[features_df['condition'] == 'Control']['neural_flexibility']
            alcoholic_flex = features_df[features_df['condition'] == 'Alcoholic']['neural_flexibility']
            
            data_to_plot = [control_flex, alcoholic_flex]
            box_plot = ax5.boxplot(data_to_plot, labels=['Control', 'Alcoholic'], patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightgreen')
            box_plot['boxes'][1].set_facecolor('orange')
            
            ax5.set_title('Neural Flexibility')
            ax5.set_ylabel('Flexibility Index')
        else:
            ax5.hist(features_df['neural_flexibility'], bins=15, alpha=0.7)
            ax5.set_title('Neural Flexibility Distribution')
        
        # 6. Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = f"""
REAL UCI ALCOHOLISM DATA

Total Recordings: {len(features_df)}
Subjects: {features_df['subject_id'].nunique()}

Conditions:
{features_df['condition'].value_counts().to_string()}

Data Source: UCI Machine Learning Repository
Format: 64-channel EEG, 256 Hz, 1-second recordings
Analysis: CTEntropy symbolic entropy features

STATUS: REAL CLINICAL DATA ANALYZED!
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_file = "results/real_alcoholism_entropy_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualizations saved to {plot_file}")
        
        plt.show()

def main():
    """Main REAL alcoholism analysis pipeline"""
    
    analyzer = RealAlcoholismCTEntropyAnalyzer()
    
    # Analyze REAL alcoholism data
    features_df = analyzer.analyze_real_alcoholism_data()
    
    if features_df is None:
        return
    
    # Train detector (or analyze patterns)
    results = analyzer.train_real_alcoholism_detector(features_df)
    
    # Statistical analysis
    analyzer.statistical_analysis(features_df)
    
    # Create visualizations
    analyzer.create_real_alcoholism_visualization(features_df, results)
    
    print("\n🎯 REAL ALCOHOLISM ANALYSIS COMPLETE!")
    print("=" * 50)
    print("✅ CTEntropy successfully analyzed REAL UCI alcoholism data")
    print("✅ Entropy signatures extracted from actual patient recordings")
    print("✅ This is REAL clinical validation - not synthetic!")
    
    return analyzer, features_df, results

if __name__ == "__main__":
    analyzer, features_df, results = main()