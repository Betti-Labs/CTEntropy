"""
UCI Alcoholism EEG Dataset Loader
Load real alcoholism vs control EEG data for CTEntropy analysis
"""

import os
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
import struct
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class UCIAlcoholismLoader:
    """Load UCI EEG Database for alcoholism detection"""
    
    def __init__(self, data_dir: str = "alcoholism-eeg-real"):
        """
        Initialize UCI alcoholism EEG loader
        
        Args:
            data_dir: Path to alcoholism-eeg-real dataset directory
        """
        self.data_dir = Path(data_dir)
        self.subjects = self._discover_subjects()
        logger.info(f"Found {len(self.subjects)} subjects in {data_dir}")
    
    def _discover_subjects(self) -> List[Dict]:
        """Auto-discover all available subjects"""
        if not self.data_dir.exists():
            logger.warning(f"Dataset directory {self.data_dir} not found")
            return []
        
        subjects = []
        
        # Look for extracted subject directories
        for subject_dir in self.data_dir.glob("co*"):
            if subject_dir.is_dir():
                subject_id = subject_dir.name
                
                # Determine condition from subject ID
                if subject_id.startswith('co2a') or subject_id.startswith('co3a'):
                    condition = 'Alcoholic'
                elif subject_id.startswith('co2c') or subject_id.startswith('co3c'):
                    condition = 'Control'
                else:
                    condition = 'Unknown'
                
                subjects.append({
                    'subject_id': subject_id,
                    'condition': condition,
                    'path': subject_dir
                })
        
        return subjects
    
    def _read_rd_file(self, rd_file_path: Path) -> Optional[np.ndarray]:
        """
        Read a .rd.gz file and extract EEG data
        
        The UCI EEG format is:
        - 64 channels
        - 256 Hz sampling rate
        - 1 second of data per file
        - Binary format with specific structure
        """
        try:
            with gzip.open(rd_file_path, 'rb') as f:
                # Read the binary data
                data = f.read()
                
                # UCI EEG format: 64 channels * 256 samples * 4 bytes (float32)
                expected_size = 64 * 256 * 4
                
                if len(data) < expected_size:
                    logger.warning(f"File {rd_file_path} too small: {len(data)} bytes")
                    return None
                
                # Unpack as float32 values
                values = struct.unpack(f'<{64*256}f', data[:expected_size])
                
                # Reshape to (channels, samples)
                eeg_data = np.array(values).reshape(64, 256)
                
                # Return first channel for entropy analysis
                return eeg_data[0, :]
                
        except Exception as e:
            logger.error(f"Failed to read {rd_file_path}: {e}")
            return None
    
    def load_subject_data(self, subject_info: Dict, max_files: int = 10) -> List[Dict]:
        """
        Load EEG data for a subject
        
        Args:
            subject_info: Subject information dictionary
            max_files: Maximum number of files to load per subject
            
        Returns:
            List of dictionaries with EEG data and metadata
        """
        subject_id = subject_info['subject_id']
        condition = subject_info['condition']
        subject_path = subject_info['path']
        
        # Get all .rd.gz files for this subject
        rd_files = list(subject_path.glob("*.rd.*.gz"))
        rd_files = sorted(rd_files)[:max_files]  # Limit files
        
        subject_data = []
        
        for rd_file in rd_files:
            logger.info(f"Loading {rd_file.name}...")
            
            # Read EEG data
            signal = self._read_rd_file(rd_file)
            if signal is None:
                continue
            
            # Store data with metadata
            data_dict = {
                'subject_id': subject_id,
                'condition': condition,
                'file': rd_file.name,
                'signal': signal,
                'sampling_rate': 256.0,  # UCI dataset is 256 Hz
                'duration': len(signal) / 256.0,
                'n_channels': 64,  # UCI dataset has 64 channels
                'is_alcoholic': 1 if condition == 'Alcoholic' else 0
            }
            
            subject_data.append(data_dict)
        
        return subject_data
    
    def load_all_subjects(self, max_subjects: int = None, max_files_per_subject: int = 5) -> pd.DataFrame:
        """
        Load EEG data for all available subjects
        
        Args:
            max_subjects: Maximum number of subjects to load
            max_files_per_subject: Maximum files per subject
            
        Returns:
            DataFrame with all EEG data and metadata
        """
        subjects_to_load = self.subjects[:max_subjects] if max_subjects else self.subjects
        
        all_data = []
        for subject_info in subjects_to_load:
            logger.info(f"Processing subject {subject_info['subject_id']}...")
            subject_data = self.load_subject_data(subject_info, max_files_per_subject)
            all_data.extend(subject_data)
        
        # Convert to DataFrame (without raw signals)
        df_data = []
        for data in all_data:
            df_data.append({
                'subject_id': data['subject_id'],
                'condition': data['condition'],
                'file': data['file'],
                'sampling_rate': data['sampling_rate'],
                'duration': data['duration'],
                'n_channels': data['n_channels'],
                'signal_length': len(data['signal']),
                'is_alcoholic': data['is_alcoholic']
            })
        
        df = pd.DataFrame(df_data)
        
        # Store raw signals separately
        self.raw_signals = {f"{d['subject_id']}_{d['file']}": d['signal'] for d in all_data}
        
        return df
    
    def get_signal(self, subject_id: str, file: str) -> Optional[np.ndarray]:
        """Get raw signal for a specific subject and file"""
        key = f"{subject_id}_{file}"
        return self.raw_signals.get(key)
    
    def extract_more_subjects(self, num_subjects: int = 5):
        """Extract more subjects from the tar.gz files"""
        
        eeg_full_dir = self.data_dir / "eeg_full"
        if not eeg_full_dir.exists():
            logger.error("eeg_full directory not found")
            return
        
        # Get tar.gz files we haven't extracted yet
        tar_files = list(eeg_full_dir.glob("*.tar.gz"))
        
        extracted_count = 0
        for tar_file in tar_files:
            if extracted_count >= num_subjects:
                break
                
            subject_id = tar_file.stem.replace('.tar', '')
            subject_dir = self.data_dir / subject_id
            
            if subject_dir.exists():
                continue  # Already extracted
            
            logger.info(f"Extracting {subject_id}...")
            
            try:
                import tarfile
                with tarfile.open(tar_file, 'r:gz') as tar:
                    tar.extractall(path=self.data_dir)
                extracted_count += 1
            except Exception as e:
                logger.error(f"Failed to extract {tar_file}: {e}")
        
        # Refresh subjects list
        self.subjects = self._discover_subjects()
        logger.info(f"Now have {len(self.subjects)} subjects available")


def test_uci_alcoholism_loader():
    """Test the UCI alcoholism loader"""
    loader = UCIAlcoholismLoader()
    
    print("🍷 Testing UCI Alcoholism EEG Loader...")
    print(f"Available subjects: {len(loader.subjects)}")
    
    if loader.subjects:
        # Show subject breakdown
        alcoholic_count = sum(1 for s in loader.subjects if s['condition'] == 'Alcoholic')
        control_count = sum(1 for s in loader.subjects if s['condition'] == 'Control')
        
        print(f"  Alcoholic subjects: {alcoholic_count}")
        print(f"  Control subjects: {control_count}")
        
        # Test loading first subject
        subject_info = loader.subjects[0]
        print(f"\nTesting with subject {subject_info['subject_id']} ({subject_info['condition']}):")
        
        subject_data = loader.load_subject_data(subject_info, max_files=3)
        
        for data in subject_data:
            print(f"  File: {data['file']}")
            print(f"  Sampling rate: {data['sampling_rate']} Hz")
            print(f"  Duration: {data['duration']:.1f} seconds")
            print(f"  Signal shape: {data['signal'].shape}")
            print(f"  Condition: {data['condition']}")
            print()
        
        # Extract more subjects if we need them
        if len(loader.subjects) < 10:
            print("Extracting more subjects...")
            loader.extract_more_subjects(num_subjects=8)
            
        return loader
    else:
        print("❌ No subjects found. Make sure dataset is extracted properly.")
        return None


if __name__ == "__main__":
    test_uci_alcoholism_loader()