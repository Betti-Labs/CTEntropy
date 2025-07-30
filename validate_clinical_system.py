#!/usr/bin/env python3
"""
Clinical System Validation Script
Tests the CTEntropy clinical system on real datasets to verify:
1. Correct detection of actual conditions
2. No false positives on healthy subjects
3. Consistent performance across different datasets
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import our clinical system and data loaders
from clinical_ctentropy_system import ClinicalCTEntropySystem
from ctentropy_platform.data.physionet_loader import PhysioNetEEGLoader
from ctentropy_platform.data.uci_alcoholism_loader import UCIAlcoholismLoader
from ctentropy_platform.data.clinical_loader import ClinicalEEGLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClinicalSystemValidator:
    """Comprehensive validation of the clinical CTEntropy system"""
    
    def __init__(self):
        self.clinical_system = ClinicalCTEntropySystem(
            facility_name="CTEntropy Validation Lab",
            physician_name="Dr. Validation"
        )
        self.results = {}
        
    def validate_on_physionet_healthy(self, max_subjects=10):
        """Test on PhysioNet healthy subjects - should detect as Healthy"""
        logger.info("🧠 Testing on PhysioNet healthy subjects...")
        
        try:
            loader = PhysioNetEEGLoader()
            subjects_df = loader.load_all_subjects(max_subjects=max_subjects, max_files_per_subject=3)
            
            predictions = []
            confidences = []
            ground_truth = []
            
            for idx, row in subjects_df.iterrows():
                try:
                    signal = loader.get_signal(row['subject'], row['file'])
                    if signal is not None and len(signal) > 1000:
                        
                        # Analyze with clinical system
                        result = self.clinical_system.analyze_patient_eeg(
                            eeg_signal=signal,
                            sampling_rate=row['sampling_rate'],
                            patient_id=f"PHYSIONET_{row['subject']}_{row['file']}",
                            user_id="VALIDATOR"
                        )
                        
                        if result['status'] == 'success':
                            predictions.append(result['diagnosis']['condition'])
                            confidences.append(result['diagnosis']['confidence'])
                            ground_truth.append('Healthy')
                            
                            logger.info(f"Subject {row['subject']}: {result['diagnosis']['condition']} ({result['diagnosis']['confidence']:.1f}%)")
                        
                except Exception as e:
                    logger.warning(f"Failed to process PhysioNet subject {row['subject']}: {e}")
                    continue
            
            self.results['physionet_healthy'] = {
                'predictions': predictions,
                'confidences': confidences,
                'ground_truth': ground_truth,
                'accuracy': accuracy_score(ground_truth, predictions) if predictions else 0.0
            }
            
            logger.info(f"PhysioNet Healthy: {len(predictions)} subjects processed")
            logger.info(f"Accuracy: {self.results['physionet_healthy']['accuracy']:.2%}")
            
        except Exception as e:
            logger.error(f"PhysioNet validation failed: {e}")
            self.results['physionet_healthy'] = {'error': str(e)}
    
    def validate_on_uci_alcoholism(self, max_subjects=10):
        """Test on UCI alcoholism dataset - should detect alcoholism vs healthy"""
        logger.info("🍺 Testing on UCI alcoholism dataset...")
        
        try:
            loader = UCIAlcoholismLoader()
            subjects_df = loader.load_all_subjects(max_subjects=max_subjects, max_files_per_subject=3)
            
            predictions = []
            confidences = []
            ground_truth = []
            
            for idx, row in subjects_df.iterrows():
                try:
                    signal = loader.get_signal(row['subject_id'], row['file'])
                    if signal is not None and len(signal) > 1000:
                        
                        # Analyze with clinical system
                        result = self.clinical_system.analyze_patient_eeg(
                            eeg_signal=signal,
                            sampling_rate=row['sampling_rate'],
                            patient_id=f"UCI_{row['subject_id']}_{row['file']}",
                            user_id="VALIDATOR"
                        )
                        
                        if result['status'] == 'success':
                            predictions.append(result['diagnosis']['condition'])
                            confidences.append(result['diagnosis']['confidence'])
                            ground_truth.append(row['condition'])
                            
                            logger.info(f"Subject {row['subject_id']} ({row['condition']}): {result['diagnosis']['condition']} ({result['diagnosis']['confidence']:.1f}%)")
                        
                except Exception as e:
                    logger.warning(f"Failed to process UCI subject {row['subject_id']}: {e}")
                    continue
            
            self.results['uci_alcoholism'] = {
                'predictions': predictions,
                'confidences': confidences,
                'ground_truth': ground_truth,
                'accuracy': accuracy_score(ground_truth, predictions) if predictions else 0.0
            }
            
            logger.info(f"UCI Alcoholism: {len(predictions)} subjects processed")
            logger.info(f"Accuracy: {self.results['uci_alcoholism']['accuracy']:.2%}")
            
        except Exception as e:
            logger.error(f"UCI alcoholism validation failed: {e}")
            self.results['uci_alcoholism'] = {'error': str(e)}
    
    def validate_on_epilepsy_data(self, max_subjects=5):
        """Test on epilepsy dataset - should detect epilepsy"""
        logger.info("⚡ Testing on epilepsy dataset...")
        
        try:
            loader = ClinicalEEGLoader("chb-mit")
            epilepsy_df = loader.load_chb_mit_epilepsy()
            
            predictions = []
            confidences = []
            ground_truth = []
            
            count = 0
            for idx, row in epilepsy_df.iterrows():
                if count >= max_subjects:
                    break
                    
                try:
                    if len(row['signal']) > 1000:
                        
                        # Analyze with clinical system
                        result = self.clinical_system.analyze_patient_eeg(
                            eeg_signal=row['signal'],
                            sampling_rate=row['sampling_rate'],
                            patient_id=f"EPILEPSY_{row['patient']}_{idx}",
                            user_id="VALIDATOR"
                        )
                        
                        if result['status'] == 'success':
                            predictions.append(result['diagnosis']['condition'])
                            confidences.append(result['diagnosis']['confidence'])
                            ground_truth.append('Epilepsy')
                            
                            logger.info(f"Epilepsy patient {row['patient']}: {result['diagnosis']['condition']} ({result['diagnosis']['confidence']:.1f}%)")
                            count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process epilepsy patient {row['patient']}: {e}")
                    continue
            
            self.results['epilepsy'] = {
                'predictions': predictions,
                'confidences': confidences,
                'ground_truth': ground_truth,
                'accuracy': accuracy_score(ground_truth, predictions) if predictions else 0.0
            }
            
            logger.info(f"Epilepsy: {len(predictions)} subjects processed")
            logger.info(f"Accuracy: {self.results['epilepsy']['accuracy']:.2%}")
            
        except Exception as e:
            logger.error(f"Epilepsy validation failed: {e}")
            self.results['epilepsy'] = {'error': str(e)}
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        logger.info("📊 Generating validation report...")
        
        # Create validation summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_datasets_tested': len([k for k, v in self.results.items() if 'error' not in v]),
            'overall_performance': {}
        }
        
        # Calculate overall metrics
        all_predictions = []
        all_ground_truth = []
        all_confidences = []
        
        for dataset_name, results in self.results.items():
            if 'error' not in results and 'predictions' in results:
                all_predictions.extend(results['predictions'])
                all_ground_truth.extend(results['ground_truth'])
                all_confidences.extend(results['confidences'])
                
                summary['overall_performance'][dataset_name] = {
                    'samples': len(results['predictions']),
                    'accuracy': results['accuracy'],
                    'avg_confidence': np.mean(results['confidences']) if results['confidences'] else 0.0
                }
        
        if all_predictions:
            summary['overall_performance']['combined'] = {
                'total_samples': len(all_predictions),
                'overall_accuracy': accuracy_score(all_ground_truth, all_predictions),
                'avg_confidence': np.mean(all_confidences)
            }
        
        # Generate detailed report
        report_lines = [
            "# CTEntropy Clinical System Validation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"- Total datasets tested: {summary['total_datasets_tested']}",
            f"- Total samples analyzed: {len(all_predictions)}",
            f"- Overall accuracy: {summary['overall_performance'].get('combined', {}).get('overall_accuracy', 0):.2%}",
            f"- Average confidence: {summary['overall_performance'].get('combined', {}).get('avg_confidence', 0):.1f}%",
            "",
            "## Dataset-Specific Results",
            ""
        ]
        
        for dataset_name, perf in summary['overall_performance'].items():
            if dataset_name != 'combined':
                report_lines.extend([
                    f"### {dataset_name.replace('_', ' ').title()}",
                    f"- Samples: {perf['samples']}",
                    f"- Accuracy: {perf['accuracy']:.2%}",
                    f"- Avg Confidence: {perf['avg_confidence']:.1f}%",
                    ""
                ])
        
        # Add detailed classification report if we have predictions
        if all_predictions:
            report_lines.extend([
                "## Detailed Classification Report",
                "```",
                classification_report(all_ground_truth, all_predictions),
                "```",
                ""
            ])
        
        # Save report
        report_path = f"clinical_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Validation report saved: {report_path}")
        return report_path, summary
    
    def create_validation_plots(self):
        """Create visualization plots for validation results"""
        logger.info("📈 Creating validation plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CTEntropy Clinical System Validation Results', fontsize=16)
        
        # Plot 1: Accuracy by dataset
        ax1 = axes[0, 0]
        datasets = []
        accuracies = []
        
        for dataset_name, results in self.results.items():
            if 'error' not in results and 'accuracy' in results:
                datasets.append(dataset_name.replace('_', '\n'))
                accuracies.append(results['accuracy'])
        
        if datasets:
            bars = ax1.bar(datasets, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'][:len(datasets)])
            ax1.set_title('Accuracy by Dataset')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            
            # Add accuracy labels on bars
            for bar, acc in zip(bars, accuracies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.1%}', ha='center', va='bottom')
        
        # Plot 2: Confidence distribution
        ax2 = axes[0, 1]
        all_confidences = []
        confidence_labels = []
        
        for dataset_name, results in self.results.items():
            if 'error' not in results and 'confidences' in results:
                all_confidences.extend(results['confidences'])
                confidence_labels.extend([dataset_name] * len(results['confidences']))
        
        if all_confidences:
            ax2.hist(all_confidences, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            ax2.set_title('Confidence Score Distribution')
            ax2.set_xlabel('Confidence (%)')
            ax2.set_ylabel('Frequency')
            ax2.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_confidences):.1f}%')
            ax2.legend()
        
        # Plot 3: Confusion matrix (if we have enough data)
        ax3 = axes[1, 0]
        all_predictions = []
        all_ground_truth = []
        
        for dataset_name, results in self.results.items():
            if 'error' not in results and 'predictions' in results:
                all_predictions.extend(results['predictions'])
                all_ground_truth.extend(results['ground_truth'])
        
        if all_predictions:
            unique_labels = sorted(list(set(all_ground_truth + all_predictions)))
            cm = confusion_matrix(all_ground_truth, all_predictions, labels=unique_labels)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=unique_labels, yticklabels=unique_labels, ax=ax3)
            ax3.set_title('Confusion Matrix')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
        
        # Plot 4: Sample counts by dataset
        ax4 = axes[1, 1]
        sample_counts = []
        dataset_names = []
        
        for dataset_name, results in self.results.items():
            if 'error' not in results and 'predictions' in results:
                dataset_names.append(dataset_name.replace('_', '\n'))
                sample_counts.append(len(results['predictions']))
        
        if dataset_names:
            bars = ax4.bar(dataset_names, sample_counts, color=['orange', 'purple', 'green'][:len(dataset_names)])
            ax4.set_title('Sample Count by Dataset')
            ax4.set_ylabel('Number of Samples')
            
            # Add count labels on bars
            for bar, count in zip(bars, sample_counts):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = f"clinical_validation_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Validation plots saved: {plot_path}")
        return plot_path

def main():
    """Run comprehensive clinical system validation"""
    print("🏥 CTEntropy Clinical System Validation")
    print("=" * 50)
    
    validator = ClinicalSystemValidator()
    
    # Test on different datasets
    print("\n🧪 Running validation tests...")
    
    # Test 1: PhysioNet healthy subjects
    validator.validate_on_physionet_healthy(max_subjects=8)
    
    # Test 2: UCI alcoholism dataset
    validator.validate_on_uci_alcoholism(max_subjects=8)
    
    # Test 3: Epilepsy dataset
    validator.validate_on_epilepsy_data(max_subjects=5)
    
    # Generate comprehensive report
    print("\n📊 Generating validation report...")
    report_path, summary = validator.generate_validation_report()
    
    # Create visualization plots
    plot_path = validator.create_validation_plots()
    
    # Print summary
    print("\n✅ Validation Complete!")
    print(f"📄 Report: {report_path}")
    print(f"📈 Plots: {plot_path}")
    
    if 'combined' in summary['overall_performance']:
        combined = summary['overall_performance']['combined']
        print(f"\n🎯 Overall Results:")
        print(f"   Total Samples: {combined['total_samples']}")
        print(f"   Overall Accuracy: {combined['overall_accuracy']:.2%}")
        print(f"   Average Confidence: {combined['avg_confidence']:.1f}%")
    
    print("\n🏥 Clinical validation completed successfully!")

if __name__ == "__main__":
    main()