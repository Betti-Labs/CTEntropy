"""
Real neurological data loaders for clinical datasets.

This module provides loaders for free, open-source neurological datasets
including PhysioNet, OpenNeuro, and other clinical data sources.
"""

import os
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
from dataclasses import dataclass
from enum import Enum

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("MNE-Python not available. Install with: pip install mne")

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    warnings.warn("WFDB not available. Install with: pip install wfdb")


class DatasetType(Enum):
    """Types of neurological datasets."""
    HEALTHY = "healthy"
    TBI = "tbi"  # Traumatic Brain Injury (includes CTE-like patterns)
    ALZHEIMERS = "alzheimers"
    DEPRESSION = "depression"
    SEIZURE = "seizure"  # Can show abnormal entropy patterns
    SLEEP_DISORDER = "sleep_disorder"  # Depression-related


@dataclass
class EEGRecord:
    """Container for EEG data record."""
    data: np.ndarray  # Shape: (n_channels, n_samples)
    sampling_rate: float
    channels: List[str]
    subject_id: str
    condition: DatasetType
    metadata: Dict
    duration: float


class PhysioNetLoader:
    """
    Loader for PhysioNet datasets.
    
    Downloads and processes EEG data from PhysioNet's free databases.
    """
    
    # PhysioNet dataset configurations
    DATASETS = {
        'eeg-motor-imagery': {
            'url': 'https://physionet.org/files/eegmmidb/1.0.0/',
            'description': 'EEG Motor Movement/Imagery Dataset',
            'subjects': 109,
            'condition': DatasetType.HEALTHY,
            'sampling_rate': 160,
            'channels': ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6', 'Fp1', 'Fpz', 'Fp2', 'Af7', 'Af3', 'Afz', 'Af4', 'Af8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'Ft7', 'Ft8', 'T7', 'T8', 'T9', 'T10', 'Tp7', 'Tp8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8', 'O1', 'Oz', 'O2', 'Iz']
        },
        'chb-mit': {
            'url': 'https://physionet.org/files/chbmit/1.0.0/',
            'description': 'CHB-MIT Scalp EEG Database (Seizure)',
            'subjects': 24,
            'condition': DatasetType.SEIZURE,
            'sampling_rate': 256,
            'channels': ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']
        },
        'sleep-edf': {
            'url': 'https://physionet.org/files/sleep-edfx/1.0.0/',
            'description': 'Sleep-EDF Database (Sleep Disorders)',
            'subjects': 197,
            'condition': DatasetType.SLEEP_DISORDER,
            'sampling_rate': 100,
            'channels': ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal']
        }
    }
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize PhysioNet loader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        if not WFDB_AVAILABLE:
            raise ImportError("WFDB package required. Install with: pip install wfdb")
    
    def list_available_datasets(self) -> Dict:
        """List all available PhysioNet datasets."""
        return self.DATASETS
    
    def download_dataset(self, dataset_name: str, max_subjects: int = 5) -> bool:
        """
        Download a PhysioNet dataset.
        
        Args:
            dataset_name: Name of dataset to download
            max_subjects: Maximum number of subjects to download (for testing)
            
        Returns:
            True if successful
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_info = self.DATASETS[dataset_name]
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"📥 Downloading {dataset_info['description']}...")
        print(f"   Subjects: {min(max_subjects, dataset_info['subjects'])}")
        print(f"   Condition: {dataset_info['condition'].value}")
        
        try:
            if dataset_name == 'eeg-motor-imagery':
                return self._download_motor_imagery(dataset_dir, max_subjects)
            elif dataset_name == 'chb-mit':
                return self._download_chb_mit(dataset_dir, max_subjects)
            elif dataset_name == 'sleep-edf':
                return self._download_sleep_edf(dataset_dir, max_subjects)
            else:
                raise NotImplementedError(f"Download not implemented for {dataset_name}")
                
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            return False
    
    def _download_motor_imagery(self, dataset_dir: Path, max_subjects: int) -> bool:
        """Download EEG Motor Movement/Imagery dataset."""
        try:
            # Download a few subjects for testing
            for subject_id in range(1, min(max_subjects + 1, 6)):  # Subjects S001-S005
                subject_str = f"S{subject_id:03d}"
                print(f"  Downloading subject {subject_str}...")
                
                # Download baseline recording (eyes open)
                record_name = f"{subject_str}R01"  # Eyes open, rest
                try:
                    record = wfdb.rdrecord(f'eegmmidb/{record_name}', 
                                         pn_dir='eegmmidb/1.0.0/')
                    
                    # Save to local directory
                    output_path = dataset_dir / f"{record_name}"
                    np.save(output_path.with_suffix('.npy'), record.p_signal)
                    
                    # Save metadata
                    metadata = {
                        'subject_id': subject_str,
                        'record_name': record_name,
                        'sampling_rate': record.fs,
                        'channels': record.sig_name,
                        'duration': len(record.p_signal) / record.fs,
                        'condition': 'healthy_baseline'
                    }
                    
                    pd.Series(metadata).to_json(output_path.with_suffix('.json'))
                    
                except Exception as e:
                    print(f"    ⚠️  Failed to download {record_name}: {e}")
                    continue
                    
            return True
            
        except Exception as e:
            print(f"❌ Motor imagery download failed: {e}")
            return False
    
    def _download_chb_mit(self, dataset_dir: Path, max_subjects: int) -> bool:
        """Download CHB-MIT seizure dataset."""
        try:
            # Download a few subjects
            for subject_id in range(1, min(max_subjects + 1, 4)):
                subject_str = f"chb{subject_id:02d}"
                print(f"  Downloading subject {subject_str}...")
                
                try:
                    # Download first recording for each subject
                    record_name = f"{subject_str}/{subject_str}_01"
                    record = wfdb.rdrecord(record_name, pn_dir='chbmit/1.0.0/')
                    
                    # Save data
                    output_path = dataset_dir / f"{subject_str}_01"
                    np.save(output_path.with_suffix('.npy'), record.p_signal)
                    
                    # Save metadata
                    metadata = {
                        'subject_id': subject_str,
                        'record_name': record_name,
                        'sampling_rate': record.fs,
                        'channels': record.sig_name,
                        'duration': len(record.p_signal) / record.fs,
                        'condition': 'seizure'
                    }
                    
                    pd.Series(metadata).to_json(output_path.with_suffix('.json'))
                    
                except Exception as e:
                    print(f"    ⚠️  Failed to download {subject_str}: {e}")
                    continue
                    
            return True
            
        except Exception as e:
            print(f"❌ CHB-MIT download failed: {e}")
            return False
    
    def _download_sleep_edf(self, dataset_dir: Path, max_subjects: int) -> bool:
        """Download Sleep-EDF dataset."""
        try:
            # Download a few subjects from sleep cassette study
            subject_ids = ['SC4001E0', 'SC4002E0', 'SC4011E0', 'SC4012E0', 'SC4021E0']
            
            for i, subject_id in enumerate(subject_ids[:max_subjects]):
                print(f"  Downloading subject {subject_id}...")
                
                try:
                    record = wfdb.rdrecord(f'sleep-edf/sleep-cassette/{subject_id}-PSG', 
                                         pn_dir='sleep-edfx/1.0.0/')
                    
                    # Save data
                    output_path = dataset_dir / subject_id
                    np.save(output_path.with_suffix('.npy'), record.p_signal)
                    
                    # Save metadata
                    metadata = {
                        'subject_id': subject_id,
                        'record_name': f"{subject_id}-PSG",
                        'sampling_rate': record.fs,
                        'channels': record.sig_name,
                        'duration': len(record.p_signal) / record.fs,
                        'condition': 'sleep_disorder'
                    }
                    
                    pd.Series(metadata).to_json(output_path.with_suffix('.json'))
                    
                except Exception as e:
                    print(f"    ⚠️  Failed to download {subject_id}: {e}")
                    continue
                    
            return True
            
        except Exception as e:
            print(f"❌ Sleep-EDF download failed: {e}")
            return False
    
    def load_dataset(self, dataset_name: str) -> List[EEGRecord]:
        """
        Load a downloaded dataset.
        
        Args:
            dataset_name: Name of dataset to load
            
        Returns:
            List of EEG records
        """
        dataset_dir = self.data_dir / dataset_name
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found. Download first.")
        
        records = []
        dataset_info = self.DATASETS[dataset_name]
        
        # Find all .npy files in dataset directory
        for npy_file in dataset_dir.glob("*.npy"):
            json_file = npy_file.with_suffix('.json')
            
            if not json_file.exists():
                continue
                
            try:
                # Load data and metadata
                data = np.load(npy_file)
                metadata = pd.read_json(json_file, typ='series').to_dict()
                
                # Create EEG record
                record = EEGRecord(
                    data=data.T,  # Transpose to (channels, samples)
                    sampling_rate=metadata['sampling_rate'],
                    channels=metadata['channels'],
                    subject_id=metadata['subject_id'],
                    condition=dataset_info['condition'],
                    metadata=metadata,
                    duration=metadata['duration']
                )
                
                records.append(record)
                
            except Exception as e:
                print(f"⚠️  Failed to load {npy_file}: {e}")
                continue
        
        print(f"✅ Loaded {len(records)} records from {dataset_name}")
        return records


class EEGDataLoader:
    """
    High-level EEG data loader that combines multiple sources.
    """
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize EEG data loader."""
        self.physionet = PhysioNetLoader(data_dir)
        self.data_dir = Path(data_dir)
    
    def setup_training_data(self, max_subjects_per_dataset: int = 3) -> Dict[str, List[EEGRecord]]:
        """
        Download and prepare training data from multiple sources.
        
        Args:
            max_subjects_per_dataset: Max subjects to download per dataset
            
        Returns:
            Dictionary mapping condition names to EEG records
        """
        print("🧠 Setting up CTEntropy training data...")
        print("=" * 50)
        
        training_data = {}
        
        # Download datasets
        datasets_to_download = ['eeg-motor-imagery', 'chb-mit', 'sleep-edf']
        
        for dataset_name in datasets_to_download:
            print(f"\n📊 Processing {dataset_name}...")
            
            # Download if not exists
            if not (self.data_dir / dataset_name).exists():
                success = self.physionet.download_dataset(dataset_name, max_subjects_per_dataset)
                if not success:
                    print(f"❌ Failed to download {dataset_name}")
                    continue
            
            # Load records
            try:
                records = self.physionet.load_dataset(dataset_name)
                
                # Group by condition
                for record in records:
                    condition_name = record.condition.value
                    if condition_name not in training_data:
                        training_data[condition_name] = []
                    training_data[condition_name].append(record)
                    
            except Exception as e:
                print(f"❌ Failed to load {dataset_name}: {e}")
                continue
        
        # Print summary
        print(f"\n📈 Training Data Summary:")
        print("-" * 30)
        total_records = 0
        for condition, records in training_data.items():
            print(f"{condition.title()}: {len(records)} records")
            total_records += len(records)
        print(f"Total: {total_records} records")
        
        return training_data
    
    def get_condition_mapping(self) -> Dict[DatasetType, str]:
        """Get mapping from dataset conditions to CTEntropy conditions."""
        return {
            DatasetType.HEALTHY: 'healthy',
            DatasetType.SEIZURE: 'cte',  # Seizure patterns similar to TBI
            DatasetType.SLEEP_DISORDER: 'depression',  # Sleep disorders linked to depression
            DatasetType.TBI: 'cte',
            DatasetType.ALZHEIMERS: 'alzheimers',
            DatasetType.DEPRESSION: 'depression'
        }