"""
PhysioNet EEG Motor Movement/Imagery Dataset Loader
Loads real clinical EEG data for CTEntropy analysis
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import mne
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PhysioNetEEGLoader:
    """Load and process PhysioNet EEG Motor Movement/Imagery Dataset"""
    
    def __init__(self, data_dir: str = "eegmmidb"):
        """
        Initialize PhysioNet EEG loader
        
        Args:
            data_dir: Path to eegmmidb dataset directory
        """
        self.data_dir = Path(data_dir)
        self.subjects = self._discover_subjects()
        logger.info(f"Found {len(self.subjects)} subjects in {data_dir}")
    
    def _discover_subjects(self) -> List[str]:
        """Auto-discover all available subjects"""
        if not self.data_dir.exists():
            logger.warning(f"Dataset directory {self.data_dir} not found")
            return []
        
        subjects = []
        for subject_dir in sorted(self.data_dir.glob("S*")):
            if subject_dir.is_dir():
                subjects.append(subject_dir.name)
        
        return subjects
    
    def get_subject_files(self, subject: str) -> List[Path]:
        """Get all EDF files for a subject"""
        subject_dir = self.data_dir / subject
        if not subject_dir.exists():
            return []
        
        edf_files = list(subject_dir.glob("*.edf"))
        return sorted(edf_files)
    
    def load_edf_file(self, edf_path: Path) -> Optional[mne.io.Raw]:
        """Load a single EDF file using MNE"""
        try:
            # Load EDF file with MNE
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            
            # Basic preprocessing
            raw.filter(l_freq=1.0, h_freq=50.0, verbose=False)  # Bandpass filter
            raw.notch_filter(freqs=60, verbose=False)  # Remove 60Hz line noise
            
            return raw
        except Exception as e:
            logger.error(f"Failed to load {edf_path}: {e}")
            return None
    
    def extract_eeg_data(self, raw: mne.io.Raw, duration: float = 60.0) -> Tuple[np.ndarray, float]:
        """
        Extract EEG data array from MNE Raw object
        
        Args:
            raw: MNE Raw object
            duration: Duration in seconds to extract
            
        Returns:
            Tuple of (data_array, sampling_rate)
        """
        # Get sampling rate
        sfreq = raw.info['sfreq']
        
        # Calculate samples for desired duration
        n_samples = int(duration * sfreq)
        
        # Get EEG data (exclude non-EEG channels)
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        data, _ = raw[eeg_picks, :n_samples]
        
        # Use first channel for entropy analysis (can be modified)
        signal = data[0, :]
        
        return signal, sfreq
    
    def load_subject_data(self, subject: str, max_files: int = 5) -> List[Dict]:
        """
        Load EEG data for a subject
        
        Args:
            subject: Subject ID (e.g., 'S001')
            max_files: Maximum number of files to load per subject
            
        Returns:
            List of dictionaries with EEG data and metadata
        """
        edf_files = self.get_subject_files(subject)
        if not edf_files:
            logger.warning(f"No EDF files found for subject {subject}")
            return []
        
        # Limit number of files to process
        edf_files = edf_files[:max_files]
        
        subject_data = []
        for edf_file in edf_files:
            logger.info(f"Loading {edf_file.name}...")
            
            # Load EDF file
            raw = self.load_edf_file(edf_file)
            if raw is None:
                continue
            
            # Extract EEG signal
            signal, sfreq = self.extract_eeg_data(raw)
            
            # Store data with metadata
            data_dict = {
                'subject': subject,
                'file': edf_file.name,
                'signal': signal,
                'sampling_rate': sfreq,
                'duration': len(signal) / sfreq,
                'n_channels': len(raw.ch_names),
                'channel_names': raw.ch_names
            }
            
            subject_data.append(data_dict)
        
        return subject_data
    
    def load_all_subjects(self, max_subjects: int = None, max_files_per_subject: int = 3) -> pd.DataFrame:
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
        for subject in subjects_to_load:
            logger.info(f"Processing subject {subject}...")
            subject_data = self.load_subject_data(subject, max_files_per_subject)
            all_data.extend(subject_data)
        
        # Convert to DataFrame
        df_data = []
        for data in all_data:
            df_data.append({
                'subject': data['subject'],
                'file': data['file'],
                'sampling_rate': data['sampling_rate'],
                'duration': data['duration'],
                'n_channels': data['n_channels'],
                'signal_length': len(data['signal'])
            })
        
        df = pd.DataFrame(df_data)
        
        # Store raw signals separately (too large for DataFrame)
        self.raw_signals = {f"{d['subject']}_{d['file']}": d['signal'] for d in all_data}
        
        return df
    
    def get_signal(self, subject: str, file: str) -> Optional[np.ndarray]:
        """Get raw signal for a specific subject and file"""
        key = f"{subject}_{file}"
        return self.raw_signals.get(key)


def test_physionet_loader():
    """Test the PhysioNet loader with available data"""
    loader = PhysioNetEEGLoader()
    
    print(f"Available subjects: {loader.subjects}")
    
    if loader.subjects:
        # Test loading first subject
        subject = loader.subjects[0]
        print(f"\nTesting with subject {subject}:")
        
        subject_data = loader.load_subject_data(subject, max_files=2)
        
        for data in subject_data:
            print(f"  File: {data['file']}")
            print(f"  Sampling rate: {data['sampling_rate']} Hz")
            print(f"  Duration: {data['duration']:.1f} seconds")
            print(f"  Signal shape: {data['signal'].shape}")
            print(f"  Channels: {len(data['channel_names'])}")
            print()


if __name__ == "__main__":
    test_physionet_loader()