"""
Test CTEntropy for Alcoholism Detection
Create research-based synthetic alcoholism patterns and test diagnostic capability
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

from ctentropy_platform.core.entropy import SymbolicEntropyCalculator
from scipy.stats import entropy as scipy_entropy
from scipy.fftpack import fft

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlcoholismEEGGenerator:
    """Generate research-based synthetic alcoholism EEG patterns"""
    
    def __init__(self, sampling_rate=160, duration=60):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.n_samples = int(sampling_rate * duration)
        
    def generate_healthy_control_eeg(self, n_subjects=45):
        """Generate healthy control EEG patterns"""
        
        signals = []
        for i in range(n_subjects):
            # Healthy brain: complex, flexible patterns
            # Base signal with multiple frequency components
            t = np.linspace(0, self.duration, self.n_samples)
            
            # Alpha rhythm (8-12 Hz) - dominant in healthy resting state
            alpha = 0.8 * np.sin(2 * np.pi * 10 * t + np.random.random() * 2 * np.pi)
            
            # Beta rhythm (13-30 Hz) - normal cognitive activity
            beta = 0.4 * np.sin(2 * np.pi * 20 * t + np.random.random() * 2 * np.pi)
            
            # Theta rhythm (4-8 Hz) - normal levels
            theta = 0.3 * np.sin(2 * np.pi * 6 * t + np.random.random() * 2 * np.pi)
            
            # Add complexity and variability (healthy neural flexibility)
            complexity = 0.2 * np.random.randn(self.n_samples)
            
            # Combine components
            signal = alpha + beta + theta + complexity
            
            # Add realistic noise
            noise = 0.1 * np.random.randn(self.n_samples)
            signal += noise
            
            signals.append({
                'subject_id': f'CTRL_{i+1:03d}',
                'condition': 'Control',
                'signal': signal,
                'sampling_rate': self.sampling_rate
            })
        
        return signals
    
    def generate_alcoholic_eeg(self, n_subjects=77):
        """Generate alcoholism EEG patterns based on research"""
        
        signals = []
        for i in range(n_subjects):
            # Alcoholic brain: reduced complexity, altered rhythms
            t = np.linspace(0, self.duration, self.n_samples)
            
            # Reduced alpha rhythm (alcoholism reduces alpha power)
            alpha = 0.5 * np.sin(2 * np.pi * 9 * t + np.random.random() * 2 * np.pi)
            
            # Increased beta activity (frontal regions in alcoholics)
            beta = 0.7 * np.sin(2 * np.pi * 22 * t + np.random.random() * 2 * np.pi)
            
            # Altered theta rhythm (different pattern in alcoholics)
            theta = 0.5 * np.sin(2 * np.pi * 5 * t + np.random.random() * 2 * np.pi)
            
            # REDUCED complexity (key difference - less neural flexibility)
            complexity = 0.1 * np.random.randn(self.n_samples)
            
            # More synchronized patterns (reduced entropy)
            synchronization = 0.3 * np.sin(2 * np.pi * 15 * t)
            
            # Combine components (more predictable patterns)
            signal = alpha + beta + theta + complexity + synchronization
            
            # Add realistic noise (but less variability)
            noise = 0.08 * np.random.randn(self.n_samples)
            signal += noise
            
            signals.append({
                'subject_id': f'ALC_{i+1:03d}',
                'condition': 'Alcoholic',
                'signal': signal,
                'sampling_rate': self.sampling_rate
            })
        
        return signals

class AlcoholismCTEntropyAnalyzer:
    """Analyze alcoholism vs control using CTEntropy"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def extract_addiction_features(self, signal, sampling_rate):
        """Extract entropy features optimized for addiction detection"""
        
        # Initialize entropy calculator
        symbolic_calc = SymbolicEntropyCalculator(window_size=50)
        
        # Multi-scale entropy analysis
        entropies_25 = SymbolicEntropyCalculator(window_size=25).calculate(signal)
        entropies_50 = symbolic_calc.calculate(signal)
        entropies_100 = SymbolicEntropyCalculator(window_size=100).calculate(signal)
        
        # Spectral entropy
        spectrum = np.abs(fft(signal))[:len(signal)//2]
        spectrum = spectrum / np.sum(spectrum)
        spectral_entropy = scipy_entropy(spectrum + 1e-9, base=2)
        
        # Addiction-specific features
        # Neural flexibility (key difference in addiction)
        neural_flexibility = np.std(entropies_50) / (np.mean(entropies_50) + 1e-9)
        
        # Synchronization measure (addiction increases synchronization)
        entropy_gradient = np.gradient(entropies_50)
        synchronization_index = 1.0 / (np.std(entropy_gradient) + 1e-9)
        
        # Frequency band analysis
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)[:len(signal)//2]
        
        # Alpha band power (8-12 Hz) - reduced in alcoholics
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        alpha_power = np.sum(spectrum[alpha_mask])
        
        # Beta band power (13-30 Hz) - increased in alcoholics
        beta_mask = (freqs >= 13) & (freqs <= 30)
        beta_power = np.sum(spectrum[beta_mask])
        
        # Alpha/Beta ratio (diagnostic marker)
        alpha_beta_ratio = alpha_power / (beta_power + 1e-9)
        
        features = {
            # Multi-scale entropy
            'entropy_25_mean': np.mean(entropies_25),
            'entropy_50_mean': np.mean(entropies_50),
            'entropy_100_mean': np.mean(entropies_100),
            
            # Entropy variability (neural flexibility)
            'entropy_std': np.std(entropies_50),
            'neural_flexibility': neural_flexibility,
            'synchronization_index': synchronization_index,
            
            # Spectral features
            'spectral_entropy': spectral_entropy,
            'alpha_power': alpha_power,
            'beta_power': beta_power,
            'alpha_beta_ratio': alpha_beta_ratio,
            
            # Temporal dynamics
            'entropy_trend': np.polyfit(range(len(entropies_50)), entropies_50, 1)[0],
            'entropy_range': np.max(entropies_50) - np.min(entropies_50)
        }
        
        return features
    
    def analyze_alcoholism_data(self):
        """Generate and analyze synthetic alcoholism data"""
        
        print("🍷 CTEntropy Alcoholism Detection Analysis")
        print("=" * 60)
        
        # Generate synthetic data
        generator = AlcoholismEEGGenerator()
        
        print("Generating synthetic EEG data based on research...")
        control_signals = generator.generate_healthy_control_eeg(n_subjects=45)
        alcoholic_signals = generator.generate_alcoholic_eeg(n_subjects=77)
        
        all_signals = control_signals + alcoholic_signals
        print(f"Generated {len(all_signals)} synthetic EEG recordings")
        print(f"  Controls: {len(control_signals)}")
        print(f"  Alcoholics: {len(alcoholic_signals)}")
        
        # Extract features
        features_list = []
        
        for signal_data in all_signals:
            subject_id = signal_data['subject_id']
            condition = signal_data['condition']
            signal = signal_data['signal']
            
            # Extract entropy features
            features = self.extract_addiction_features(signal, signal_data['sampling_rate'])
            features['subject_id'] = subject_id
            features['condition'] = condition
            features['is_alcoholic'] = 1 if condition == 'Alcoholic' else 0
            
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Save features
        features_file = "results/alcoholism_entropy_features.csv"
        Path("results").mkdir(exist_ok=True)
        features_df.to_csv(features_file, index=False)
        print(f"✅ Features saved to {features_file}")
        
        return features_df
    
    def train_alcoholism_detector(self, features_df):
        """Train alcoholism detection model"""
        
        print("\n🤖 Training Alcoholism Detection Model...")
        
        # Prepare features and labels
        feature_cols = [col for col in features_df.columns 
                       if col not in ['subject_id', 'condition', 'is_alcoholic']]
        
        X = features_df[feature_cols]
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
        print(f"Alcoholism Detection Accuracy: {accuracy:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Control', 'Alcoholic']))
        
        return {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_importance': self.model.feature_importances_,
            'feature_names': feature_cols
        }
    
    def statistical_analysis(self, features_df):
        """Perform statistical analysis between groups"""
        
        print("\n📊 Statistical Analysis: Control vs Alcoholic")
        print("=" * 50)
        
        control_data = features_df[features_df['condition'] == 'Control']
        alcoholic_data = features_df[features_df['condition'] == 'Alcoholic']
        
        # Key entropy measures
        measures = ['entropy_50_mean', 'neural_flexibility', 'spectral_entropy', 'alpha_beta_ratio']
        
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
    
    def create_alcoholism_visualization(self, features_df, results):
        """Create comprehensive alcoholism analysis visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CTEntropy Alcoholism Detection Analysis', fontsize=16, fontweight='bold')
        
        # 1. Entropy comparison
        ax1 = axes[0, 0]
        control_entropy = features_df[features_df['condition'] == 'Control']['entropy_50_mean']
        alcoholic_entropy = features_df[features_df['condition'] == 'Alcoholic']['entropy_50_mean']
        
        data_to_plot = [control_entropy, alcoholic_entropy]
        box_plot = ax1.boxplot(data_to_plot, labels=['Control', 'Alcoholic'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        
        ax1.set_title('Symbolic Entropy: Control vs Alcoholic')
        ax1.set_ylabel('Symbolic Entropy')
        
        # 2. Neural flexibility
        ax2 = axes[0, 1]
        control_flex = features_df[features_df['condition'] == 'Control']['neural_flexibility']
        alcoholic_flex = features_df[features_df['condition'] == 'Alcoholic']['neural_flexibility']
        
        data_to_plot = [control_flex, alcoholic_flex]
        box_plot = ax2.boxplot(data_to_plot, labels=['Control', 'Alcoholic'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        box_plot['boxes'][1].set_facecolor('orange')
        
        ax2.set_title('Neural Flexibility')
        ax2.set_ylabel('Flexibility Index')
        
        # 3. Alpha/Beta ratio
        ax3 = axes[0, 2]
        control_ratio = features_df[features_df['condition'] == 'Control']['alpha_beta_ratio']
        alcoholic_ratio = features_df[features_df['condition'] == 'Alcoholic']['alpha_beta_ratio']
        
        data_to_plot = [control_ratio, alcoholic_ratio]
        box_plot = ax3.boxplot(data_to_plot, labels=['Control', 'Alcoholic'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('gold')
        box_plot['boxes'][1].set_facecolor('purple')
        
        ax3.set_title('Alpha/Beta Power Ratio')
        ax3.set_ylabel('Alpha/Beta Ratio')
        
        # 4. Feature space
        ax4 = axes[1, 0]
        for condition in features_df['condition'].unique():
            subset = features_df[features_df['condition'] == condition]
            ax4.scatter(subset['entropy_50_mean'], subset['neural_flexibility'], 
                       label=condition, alpha=0.7)
        ax4.set_xlabel('Symbolic Entropy')
        ax4.set_ylabel('Neural Flexibility')
        ax4.set_title('Addiction Feature Space')
        ax4.legend()
        
        # 5. Feature importance
        ax5 = axes[1, 1]
        if 'feature_importance' in results:
            importance_df = pd.DataFrame({
                'feature': results['feature_names'],
                'importance': results['feature_importance']
            }).sort_values('importance', ascending=True)
            
            ax5.barh(importance_df['feature'][-8:], importance_df['importance'][-8:])
            ax5.set_title('Top Feature Importances')
            ax5.set_xlabel('Importance')
        
        # 6. Confusion matrix
        ax6 = axes[1, 2]
        if 'y_test' in results and 'y_pred' in results:
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', ax=ax6, cmap='Blues',
                       xticklabels=['Control', 'Alcoholic'],
                       yticklabels=['Control', 'Alcoholic'])
            ax6.set_title('Confusion Matrix')
            ax6.set_xlabel('Predicted')
            ax6.set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = "results/alcoholism_detection_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualizations saved to {plot_file}")
        
        plt.show()

def main():
    """Main alcoholism detection analysis"""
    
    analyzer = AlcoholismCTEntropyAnalyzer()
    
    # Generate and analyze data
    features_df = analyzer.analyze_alcoholism_data()
    
    # Train detector
    results = analyzer.train_alcoholism_detector(features_df)
    
    # Statistical analysis
    analyzer.statistical_analysis(features_df)
    
    # Create visualizations
    analyzer.create_alcoholism_visualization(features_df, results)
    
    print(f"\n🎯 ALCOHOLISM DETECTION ANALYSIS COMPLETE!")
    print(f"=" * 50)
    print(f"✅ CTEntropy successfully tested on synthetic alcoholism data")
    print(f"✅ Accuracy: {results['accuracy']:.1%}")
    print(f"✅ Proof of concept for addiction screening")
    print(f"✅ Ready to test on real alcoholism EEG data when available")
    
    return analyzer, features_df, results

if __name__ == "__main__":
    analyzer, features_df, results = main()