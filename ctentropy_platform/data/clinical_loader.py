"""
Clinical EEG Dataset Loader with Medical Diagnoses
Loads EEG data with known medical conditions for CTEntropy validation
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

class ClinicalEEGLoader:
    """Load clinical EEG datasets with medical diagnoses"""
    
    def __init__(self, dataset_type: str = "chb-mit"):
        """
        Initialize clinical EEG loader
        
        Args:
            dataset_type: Type of clinical dataset ('chb-mit', 'sleep-edf', etc.)
        """
        self.dataset_type = dataset_type
        self.data_dir = Path(f"{dataset_type}-epilepsy" if dataset_type == "chb-mit" else dataset_type)
        
    def load_chb_mit_epilepsy(self) -> pd.DataFrame:
        """Load CHB-MIT epilepsy dataset with seizure annotations"""
        
        if not self.data_dir.exists():
            logger.error(f"Dataset directory {self.data_dir} not found")
            return pd.DataFrame()
        
        patients = []
        for patient_dir in sorted(self.data_dir.glob("chb*")):
            if not patient_dir.is_dir():
                continue
                
            patient_id = patient_dir.name
            logger.info(f"Loading patient {patient_id}...")
            
            # Load EDF files for this patient
            edf_files = list(patient_dir.glob("*.edf"))
            
            for edf_file in edf_files[:3]:  # Limit files per patient
                try:
                    # Load EDF file
                    raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
                    
                    # Basic preprocessing
                    raw.filter(l_freq=1.0, h_freq=50.0, verbose=False)
                    raw.notch_filter(freqs=60, verbose=False)
                    
                    # Extract signal (first EEG channel)
                    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
                    if len(eeg_picks) > 0:
                        signal, _ = raw[eeg_picks[0], :int(60 * raw.info['sfreq'])]  # 60 seconds
                        
                        # Determine if this is seizure or normal
                        # CHB-MIT files with 'sz' in name contain seizures
                        has_seizure = 'sz' in edf_file.name.lower()
                        condition = 'Epilepsy_Seizure' if has_seizure else 'Epilepsy_Normal'
                        
                        patients.append({
                            'patient_id': patient_id,
                            'file': edf_file.name,
                            'condition': condition,
                            'has_medical_diagnosis': True,
                            'diagnosis': 'Epilepsy',
                            'signal': signal.flatten(),
                            'sampling_rate': raw.info['sfreq'],
                            'duration': len(signal.flatten()) / raw.info['sfreq']
                        })
                        
                except Exception as e:
                    logger.error(f"Failed to load {edf_file}: {e}")
                    continue
        
        return pd.DataFrame(patients)
    
    def create_diagnosis_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary diagnosis labels for machine learning"""
        
        # Create binary labels
        df['is_abnormal'] = df['condition'].str.contains('Seizure').astype(int)
        df['is_epilepsy'] = df['diagnosis'].str.contains('Epilepsy').astype(int)
        
        return df


def test_clinical_loader():
    """Test the clinical loader"""
    loader = ClinicalEEGLoader("chb-mit")
    
    print("Testing Clinical EEG Loader...")
    
    # Try to load epilepsy data
    df = loader.load_chb_mit_epilepsy()
    
    if len(df) > 0:
        print(f"Loaded {len(df)} clinical recordings")
        print(f"Patients: {df['patient_id'].nunique()}")
        print(f"Conditions: {df['condition'].value_counts().to_dict()}")
        print(f"Diagnoses: {df['diagnosis'].value_counts().to_dict()}")
        
        # Add diagnosis labels
        df = loader.create_diagnosis_labels(df)
        print(f"Normal vs Abnormal: {df['is_abnormal'].value_counts().to_dict()}")
        
        return df
    else:
        print("No clinical data found. Run download script first!")
        return None


if __name__ == "__main__":
    test_clinical_loader()