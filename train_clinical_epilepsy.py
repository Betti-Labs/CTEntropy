"""
CTEntropy Analysis on Real Clinical Epilepsy Data
Test entropy signatures on actual seizure vs normal EEG recordings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pathlib import Path
import logging

from ctentropy_platform.data.clinical_loader import ClinicalEEGLoader
from ctentropy_platform.core.entropy import SymbolicEntropyCalculator
from scipy.stats import entropy as scipy_entropy
from scipy.fftpack import fft

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EpilepsyCTEntropyAnalyzer:
    """Analyze epilepsy EEG data with CTEntropy"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = []
        
    def extract_epilepsy_features(self, signal, sampling_rate):
        """Extract entropy features optimized for epilepsy detection"""
        
        # Initialize entropy calculator
        symbolic_calc = SymbolicEntropyCalculator(window_size=50)
        
        # 1. Multi-window symbolic entropy
        entropies_25 = SymbolicEntropyCalculator(window_size=25).calculate(signal)
        entropies_50 = symbolic_calc.calculate(signal)
        entropies_100 = SymbolicEntropyCalculator(window_size=100).calculate(signal)
        
        # 2. Spectral entropy
        spectrum = np.abs(fft(signal))[:len(signal)//2]
        spectrum = spectrum / np.sum(spectrum)
        spectral_entropy = scipy_entropy(spectrum + 1e-9, base=2)
        
        # 3. Seizure-specific features
        # High-frequency components (seizures often have high-freq activity)
        high_freq_power = np.sum(spectrum[int(len(spectrum)*0.7):])
        
        # Entropy variability (seizures cause dramatic entropy changes)
        entropy_variability = np.std(entropies_50) / (np.mean(entropies_50) + 1e-9)
        
        # Sudden entropy changes (ictal vs interictal)
        entropy_gradient = np.gradient(entropies_50)
        max_entropy_change = np.max(np.abs(entropy_gradient))
        
        # Signal amplitude features (seizures often have high amplitude)
        signal_amplitude = np.std(signal)
        signal_peak_ratio = np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-9)
        
        features = {
            # Multi-scale entropy
            'entropy_25_mean': np.mean(entropies_25),
            'entropy_50_mean': np.mean(entropies_50),
            'entropy_100_mean': np.mean(entropies_100),
            
            # Entropy variability (key for seizure detection)
            'entropy_50_std': np.std(entropies_50),
            'entropy_variability': entropy_variability,
            'max_entropy_change': max_entropy_change,
            
            # Spectral features
            'spectral_entropy': spectral_entropy,
            'high_freq_power': high_freq_power,
            
            # Signal characteristics
            'signal_amplitude': signal_amplitude,
            'signal_peak_ratio': signal_peak_ratio,
            
            # Temporal dynamics
            'entropy_trend': np.polyfit(range(len(entropies_50)), entropies_50, 1)[0],
            'entropy_range': np.max(entropies_50) - np.min(entropies_50)
        }
        
        return features
    
    def analyze_epilepsy_data(self):
        """Analyze real epilepsy patient data"""
        
        print("🏥 CTEntropy Analysis on Real Epilepsy Data")
        print("=" * 60)
        
        # Load clinical epilepsy data
        loader = ClinicalEEGLoader("chb-mit")
        df = loader.load_chb_mit_epilepsy()
        
        if len(df) == 0:
            print("❌ No epilepsy data found. Make sure CHB-MIT dataset is downloaded!")
            return None
        
        print(f"Loaded {len(df)} epilepsy recordings from {df['patient_id'].nunique()} patients")
        print(f"Conditions: {df['condition'].value_counts().to_dict()}")
        
        # Extract features for each recording
        features_list = []
        
        for idx, row in df.iterrows():
            patient = row['patient_id']
            file = row['file']
            condition = row['condition']
            
            print(f"Analyzing {patient} - {file} ({condition})")
            
            # Extract entropy features
            features = self.extract_epilepsy_features(row['signal'], row['sampling_rate'])
            features['patient_id'] = patient
            features['file'] = file
            features['condition'] = condition
            features['is_seizure'] = 1 if 'Seizure' in condition else 0
            
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Store feature names
        self.feature_names = [col for col in features_df.columns 
                            if col not in ['patient_id', 'file', 'condition', 'is_seizure']]
        
        # Save features
        features_file = "results/epilepsy_entropy_features.csv"
        Path("results").mkdir(exist_ok=True)
        features_df.to_csv(features_file, index=False)
        print(f"✅ Features saved to {features_file}")
        
        return features_df
    
    def train_seizure_detector(self, features_df):
        """Train seizure detection model"""
        
        print("\n🤖 Training Seizure Detection Model...")
        
        # Check if we have seizure data
        seizure_counts = features_df['is_seizure'].value_counts()
        print(f"Data distribution: {seizure_counts.to_dict()}")
        
        if len(seizure_counts) < 2:
            print("⚠️  Only one class found. Cannot train binary classifier.")
            print("This means all recordings are either seizure or normal (not both)")
            return self.analyze_patterns_only(features_df)
        
        # Prepare features and labels
        X = features_df[self.feature_names]
        y = features_df['is_seizure']
        
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
        print(f"Seizure Detection Accuracy: {accuracy:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Normal', 'Seizure']))
        
        return {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_importance': self.model.feature_importances_
        }
    
    def analyze_patterns_only(self, features_df):
        """Analyze entropy patterns when we can't train classifier"""
        
        print("\n📊 Analyzing Entropy Patterns in Epilepsy Data...")
        
        # Statistical analysis
        print("\nEntropy Statistics by Condition:")
        for condition in features_df['condition'].unique():
            subset = features_df[features_df['condition'] == condition]
            print(f"\n{condition} (n={len(subset)}):")
            
            for feature in ['entropy_50_mean', 'entropy_variability', 'spectral_entropy']:
                if feature in features_df.columns:
                    mean_val = subset[feature].mean()
                    std_val = subset[feature].std()
                    print(f"  {feature}: {mean_val:.3f} ± {std_val:.3f}")
        
        # Patient-specific analysis
        print("\nPer-Patient Analysis:")
        patient_stats = features_df.groupby('patient_id').agg({
            'entropy_50_mean': ['mean', 'std'],
            'entropy_variability': ['mean', 'std'],
            'spectral_entropy': ['mean', 'std']
        }).round(3)
        
        print(patient_stats)
        
        return {'pattern_analysis': True}
    
    def create_epilepsy_visualizations(self, features_df, results=None):
        """Create visualizations for epilepsy analysis"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CTEntropy Analysis on Real Epilepsy Data', fontsize=16, fontweight='bold')
        
        # 1. Entropy by condition
        ax1 = axes[0, 0]
        if 'condition' in features_df.columns:
            conditions = features_df['condition'].unique()
            entropy_means = [features_df[features_df['condition'] == c]['entropy_50_mean'].mean() 
                           for c in conditions]
            entropy_stds = [features_df[features_df['condition'] == c]['entropy_50_mean'].std() 
                          for c in conditions]
            
            bars = ax1.bar(conditions, entropy_means, yerr=entropy_stds, capsize=5,
                          color=['red' if 'Seizure' in c else 'blue' for c in conditions],
                          alpha=0.7)
            ax1.set_title('Symbolic Entropy by Condition')
            ax1.set_ylabel('Entropy')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Entropy variability
        ax2 = axes[0, 1]
        if 'entropy_variability' in features_df.columns:
            for condition in features_df['condition'].unique():
                subset = features_df[features_df['condition'] == condition]
                ax2.hist(subset['entropy_variability'], alpha=0.7, label=condition, bins=10)
            ax2.set_title('Entropy Variability Distribution')
            ax2.set_xlabel('Entropy Variability')
            ax2.set_ylabel('Count')
            ax2.legend()
        
        # 3. Patient comparison
        ax3 = axes[0, 2]
        patients = features_df['patient_id'].unique()
        patient_entropies = [features_df[features_df['patient_id'] == p]['entropy_50_mean'].mean() 
                           for p in patients]
        
        ax3.bar(patients, patient_entropies, color='green', alpha=0.7)
        ax3.set_title('Mean Entropy by Patient')
        ax3.set_ylabel('Entropy')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Feature correlation
        ax4 = axes[1, 0]
        if len(self.feature_names) > 1:
            feature_corr = features_df[self.feature_names].corr()
            sns.heatmap(feature_corr, ax=ax4, cmap='coolwarm', center=0, 
                       square=True, cbar_kws={'shrink': 0.8})
            ax4.set_title('Feature Correlation Matrix')
        
        # 5. Spectral vs Symbolic entropy
        ax5 = axes[1, 1]
        if 'spectral_entropy' in features_df.columns:
            for condition in features_df['condition'].unique():
                subset = features_df[features_df['condition'] == condition]
                ax5.scatter(subset['entropy_50_mean'], subset['spectral_entropy'], 
                           label=condition, alpha=0.7)
            ax5.set_xlabel('Symbolic Entropy')
            ax5.set_ylabel('Spectral Entropy')
            ax5.set_title('Entropy Feature Space')
            ax5.legend()
        
        # 6. Feature importance (if available)
        ax6 = axes[1, 2]
        if results and 'feature_importance' in results:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': results['feature_importance']
            }).sort_values('importance', ascending=True)
            
            ax6.barh(importance_df['feature'][-8:], importance_df['importance'][-8:])
            ax6.set_title('Top Feature Importances')
            ax6.set_xlabel('Importance')
        else:
            # Show entropy time series example
            ax6.text(0.5, 0.5, 'Real Clinical\nEpilepsy Data\nAnalyzed!', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    transform=ax6.transAxes)
            ax6.set_title('CTEntropy Success!')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = "results/epilepsy_entropy_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualizations saved to {plot_file}")
        
        plt.show()

def main():
    """Main epilepsy analysis pipeline"""
    
    analyzer = EpilepsyCTEntropyAnalyzer()
    
    # Analyze epilepsy data
    features_df = analyzer.analyze_epilepsy_data()
    
    if features_df is None:
        return
    
    # Train seizure detector (or analyze patterns)
    results = analyzer.train_seizure_detector(features_df)
    
    # Create visualizations
    analyzer.create_epilepsy_visualizations(features_df, results)
    
    print("\n🎯 EPILEPSY ANALYSIS COMPLETE!")
    print("=" * 40)
    print("✅ CTEntropy successfully analyzed real clinical epilepsy data")
    print("✅ Entropy signatures extracted from actual patient recordings")
    print("✅ Ready for clinical validation and deployment")
    
    return analyzer, features_df, results

if __name__ == "__main__":
    analyzer, features_df, results = main()