"""
Machine Learning Classifier for CTEntropy EEG Analysis
Train models to classify neurological conditions based on entropy signatures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pathlib import Path
import logging

from ctentropy_platform.data.physionet_loader import PhysioNetEEGLoader
from ctentropy_platform.core.entropy import SymbolicEntropyCalculator
from scipy.stats import entropy as scipy_entropy
from scipy.fftpack import fft

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTEntropyMLClassifier:
    """Machine Learning classifier for CTEntropy analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42)
        }
        self.trained_models = {}
        self.feature_names = []
        
    def extract_entropy_features(self, signal, sampling_rate):
        """Extract comprehensive entropy features from EEG signal"""
        
        # Initialize entropy calculator
        symbolic_calc = SymbolicEntropyCalculator(window_size=50)
        
        # 1. Symbolic entropy features
        symbolic_entropies = symbolic_calc.calculate(signal)
        
        # 2. Spectral entropy
        spectrum = np.abs(fft(signal))[:len(signal)//2]
        spectrum = spectrum / np.sum(spectrum)
        spectral_entropy = scipy_entropy(spectrum + 1e-9, base=2)
        
        # 3. Multi-scale entropy features
        entropies_25 = SymbolicEntropyCalculator(window_size=25).calculate(signal)
        entropies_100 = SymbolicEntropyCalculator(window_size=100).calculate(signal)
        
        # 4. Statistical features of entropy time series
        features = {
            # Basic entropy measures
            'symbolic_entropy_mean': np.mean(symbolic_entropies),
            'symbolic_entropy_std': np.std(symbolic_entropies),
            'symbolic_entropy_min': np.min(symbolic_entropies),
            'symbolic_entropy_max': np.max(symbolic_entropies),
            'spectral_entropy': spectral_entropy,
            
            # Multi-scale features
            'entropy_25_mean': np.mean(entropies_25),
            'entropy_100_mean': np.mean(entropies_100),
            
            # Temporal dynamics
            'entropy_trend': np.polyfit(range(len(symbolic_entropies)), symbolic_entropies, 1)[0],
            'entropy_variability': np.std(symbolic_entropies) / np.mean(symbolic_entropies),
            
            # Collapse detection
            'entropy_drop': (np.mean(symbolic_entropies[:len(symbolic_entropies)//4]) - 
                           np.mean(symbolic_entropies[-len(symbolic_entropies)//4:])),
            
            # Signal characteristics
            'signal_std': np.std(signal),
            'signal_skewness': float(pd.Series(signal).skew()),
            'signal_kurtosis': float(pd.Series(signal).kurtosis())
        }
        
        return features
    
    def prepare_training_data(self, max_subjects=None):
        """Prepare training data from PhysioNet dataset"""
        
        print("🔄 Preparing training data...")
        
        # Load PhysioNet data
        loader = PhysioNetEEGLoader()
        df = loader.load_all_subjects(max_subjects=max_subjects, max_files_per_subject=5)
        
        print(f"Loaded {len(df)} recordings from {df['subject'].nunique()} subjects")
        
        # Extract features for each recording
        features_list = []
        labels = []
        
        for _, row in df.iterrows():
            subject = row['subject']
            file = row['file']
            
            print(f"Extracting features: {subject} - {file}")
            
            # Get signal
            signal = loader.get_signal(subject, file)
            if signal is None:
                continue
            
            # Extract entropy features
            features = self.extract_entropy_features(signal, row['sampling_rate'])
            features['subject'] = subject
            features['file'] = file
            
            features_list.append(features)
            
            # Create synthetic labels based on subject patterns
            # (In real clinical data, these would be medical diagnoses)
            labels.append(self._create_synthetic_label(subject, features))
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        features_df['label'] = labels
        
        # Store feature names
        self.feature_names = [col for col in features_df.columns 
                            if col not in ['subject', 'file', 'label']]
        
        return features_df
    
    def _create_synthetic_label(self, subject, features):
        """Create synthetic labels based on entropy patterns"""
        
        # Create labels based on entropy characteristics
        # This simulates different neurological conditions
        
        entropy_mean = features['symbolic_entropy_mean']
        entropy_std = features['symbolic_entropy_std']
        entropy_trend = features['entropy_trend']
        
        # High entropy + high variability = "Chaotic" (simulates TBI/CTE)
        if entropy_mean > 3.9 and entropy_std > 0.1:
            return 'High_Complexity'
        
        # Low entropy + stable = "Reduced" (simulates Depression)
        elif entropy_mean < 3.7 and entropy_std < 0.05:
            return 'Low_Complexity'
        
        # Declining entropy = "Degenerative" (simulates Alzheimer's)
        elif entropy_trend < -0.01:
            return 'Declining_Complexity'
        
        # Normal range
        else:
            return 'Normal_Complexity'
    
    def train_models(self, features_df):
        """Train machine learning models"""
        
        print("🤖 Training ML models...")
        
        # Prepare features and labels
        X = features_df[self.feature_names]
        y = features_df['label']
        
        print(f"Features shape: {X.shape}")
        print(f"Label distribution: {y.value_counts().to_dict()}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train models
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Test predictions
            y_pred = model.predict(X_test)
            
            results[name] = {
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'test_accuracy': model.score(X_test, y_test),
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"  CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            print(f"  Test Accuracy: {model.score(X_test, y_test):.3f}")
        
        return results, X_test, y_test
    
    def create_visualizations(self, results, features_df):
        """Create comprehensive visualizations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CTEntropy Machine Learning Analysis', fontsize=16, fontweight='bold')
        
        # 1. Feature importance (Random Forest)
        if 'RandomForest' in self.trained_models:
            rf_model = self.trained_models['RandomForest']
            importances = rf_model.feature_importances_
            
            ax1 = axes[0, 0]
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            ax1.barh(feature_importance['feature'][-10:], feature_importance['importance'][-10:])
            ax1.set_title('Top 10 Feature Importances')
            ax1.set_xlabel('Importance')
        
        # 2. Label distribution
        ax2 = axes[0, 1]
        label_counts = features_df['label'].value_counts()
        ax2.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
        ax2.set_title('Label Distribution')
        
        # 3. Entropy scatter plot
        ax3 = axes[0, 2]
        for label in features_df['label'].unique():
            subset = features_df[features_df['label'] == label]
            ax3.scatter(subset['symbolic_entropy_mean'], subset['spectral_entropy'], 
                       label=label, alpha=0.7)
        ax3.set_xlabel('Symbolic Entropy Mean')
        ax3.set_ylabel('Spectral Entropy')
        ax3.set_title('Entropy Feature Space')
        ax3.legend()
        
        # 4. Model comparison
        ax4 = axes[1, 0]
        model_names = list(results.keys())
        cv_scores = [results[name]['cv_mean'] for name in model_names]
        test_scores = [results[name]['test_accuracy'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax4.bar(x - width/2, cv_scores, width, label='CV Score', alpha=0.7)
        ax4.bar(x + width/2, test_scores, width, label='Test Score', alpha=0.7)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Model Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names)
        ax4.legend()
        
        # 5. Confusion matrix for best model
        best_model = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        ax5 = axes[1, 1]
        
        cm = confusion_matrix(results[best_model]['y_test'], results[best_model]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', ax=ax5, cmap='Blues')
        ax5.set_title(f'Confusion Matrix - {best_model}')
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('Actual')
        
        # 6. Feature correlation heatmap
        ax6 = axes[1, 2]
        feature_corr = features_df[self.feature_names].corr()
        sns.heatmap(feature_corr, ax=ax6, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        ax6.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = "results/ctentropy_ml_analysis.png"
        Path("results").mkdir(exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualizations saved to {plot_file}")
        
        plt.show()
        
        return fig

def main():
    """Main training pipeline"""
    
    print("🧠 CTEntropy Machine Learning Pipeline")
    print("=" * 50)
    
    # Initialize classifier
    classifier = CTEntropyMLClassifier()
    
    # Prepare training data
    features_df = classifier.prepare_training_data(max_subjects=10)
    
    # Save features
    features_file = "results/ctentropy_features.csv"
    features_df.to_csv(features_file, index=False)
    print(f"✅ Features saved to {features_file}")
    
    # Train models
    results, X_test, y_test = classifier.train_models(features_df)
    
    # Create visualizations
    classifier.create_visualizations(results, features_df)
    
    # Print detailed results
    print("\n📊 Detailed Results:")
    print("=" * 30)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Cross-validation: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
        print(f"  Test accuracy: {result['test_accuracy']:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(result['y_test'], result['y_pred']))
    
    return classifier, features_df, results

if __name__ == "__main__":
    classifier, features_df, results = main()