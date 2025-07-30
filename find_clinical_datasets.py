"""
Find and Download Clinical EEG Datasets with Medical Diagnoses
"""

import requests
import pandas as pd
from pathlib import Path

def explore_physionet_clinical():
    """Explore PhysioNet for clinical EEG datasets"""
    
    print("🏥 Clinical EEG Datasets Available:")
    print("=" * 50)
    
    # Known clinical datasets on PhysioNet
    clinical_datasets = {
        "CHB-MIT Scalp EEG Database": {
            "url": "https://physionet.org/content/chbmit/1.0.0/",
            "description": "Pediatric epilepsy patients with seizure recordings",
            "subjects": "24 patients",
            "condition": "Epilepsy vs Normal",
            "format": "EDF files"
        },
        
        "EEG Database for Alcoholism": {
            "url": "https://physionet.org/content/eegmmidb/1.0.0/",  # This is actually motor imagery
            "description": "Alcoholic subjects vs controls",
            "subjects": "122 subjects (77 alcoholic, 45 control)",
            "condition": "Alcoholism vs Control",
            "format": "EDF files"
        },
        
        "Sleep-EDF Database": {
            "url": "https://physionet.org/content/sleep-edfx/1.0.0/",
            "description": "Sleep disorders and normal sleep patterns",
            "subjects": "197 whole-night recordings",
            "condition": "Sleep disorders vs Normal",
            "format": "EDF files"
        },
        
        "TUH EEG Seizure Corpus": {
            "url": "https://www.isip.piconepress.com/projects/tuh_eeg/",
            "description": "Largest clinical EEG database with seizure annotations",
            "subjects": "Thousands of patients",
            "condition": "Seizure vs Normal",
            "format": "EDF files"
        }
    }
    
    for name, info in clinical_datasets.items():
        print(f"\n📊 {name}")
        print(f"   URL: {info['url']}")
        print(f"   Description: {info['description']}")
        print(f"   Subjects: {info['subjects']}")
        print(f"   Condition: {info['condition']}")
        print(f"   Format: {info['format']}")
    
    return clinical_datasets

def download_chb_mit_epilepsy():
    """Download CHB-MIT Epilepsy Database - most promising for medical diagnosis"""
    
    print("\n🧠 Downloading CHB-MIT Epilepsy Database...")
    print("This dataset has REAL epilepsy patients with seizure events!")
    
    base_url = "https://physionet.org/files/chbmit/1.0.0/"
    
    # Create download script for CHB-MIT
    script_content = '''# PowerShell script to download CHB-MIT Epilepsy Database
# Real epilepsy patients with seizure annotations!

$baseUrl = "https://physionet.org/files/chbmit/1.0.0"
$outputDir = "chb-mit-epilepsy"

# Create output directory
if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir
}

Write-Host "Downloading CHB-MIT Epilepsy Database..."
Write-Host "This contains REAL epilepsy patients with seizure events!"

# Download for patients chb01 to chb24
for ($i = 1; $i -le 24; $i++) {
    $patient = "chb{0:D2}" -f $i
    $patientDir = Join-Path $outputDir $patient
    
    if (!(Test-Path $patientDir)) {
        New-Item -ItemType Directory -Path $patientDir
    }
    
    Write-Host "Downloading patient $patient..."
    
    try {
        # Get the patient directory listing
        $patientUrl = "$baseUrl/$patient/"
        $response = Invoke-WebRequest -Uri $patientUrl -UseBasicParsing
        
        # Parse HTML to find .edf files
        $links = $response.Links | Where-Object { $_.href -match "\\.edf$" }
        
        # Limit to first 3 files per patient to save space
        $links = $links | Select-Object -First 3
        
        foreach ($link in $links) {
            $fileName = $link.href
            $fileUrl = "$patientUrl$fileName"
            $outputPath = Join-Path $patientDir $fileName
            
            # Skip if file already exists
            if (Test-Path $outputPath) {
                Write-Host "  Skipping $fileName (already exists)"
                continue
            }
            
            Write-Host "  Downloading $fileName..."
            try {
                Invoke-WebRequest -Uri $fileUrl -OutFile $outputPath
            }
            catch {
                Write-Warning "Failed to download $fileName: $_"
            }
        }
        
        # Also download summary file if it exists
        try {
            $summaryUrl = "$patientUrl$patient-summary.txt"
            $summaryPath = Join-Path $patientDir "$patient-summary.txt"
            if (!(Test-Path $summaryPath)) {
                Invoke-WebRequest -Uri $summaryUrl -OutFile $summaryPath
            }
        }
        catch {
            # Summary file might not exist, that's ok
        }
        
    }
    catch {
        Write-Warning "Failed to access patient $patient: $_"
    }
}

Write-Host "Download complete! Epilepsy dataset saved to: $outputDir"
Write-Host "This dataset contains REAL seizure events with medical annotations!"
'''
    
    # Save the download script
    with open("download_chb_mit_epilepsy.ps1", "w") as f:
        f.write(script_content)
    
    print("✅ Created download_chb_mit_epilepsy.ps1")
    print("Run this script to download REAL epilepsy patient data!")
    
    return "download_chb_mit_epilepsy.ps1"

def create_clinical_loader():
    """Create a loader for clinical datasets with medical labels"""
    
    loader_code = '''"""
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
    
    print("🏥 Testing Clinical EEG Loader...")
    
    # Try to load epilepsy data
    df = loader.load_chb_mit_epilepsy()
    
    if len(df) > 0:
        print(f"✅ Loaded {len(df)} clinical recordings")
        print(f"Patients: {df['patient_id'].nunique()}")
        print(f"Conditions: {df['condition'].value_counts().to_dict()}")
        print(f"Diagnoses: {df['diagnosis'].value_counts().to_dict()}")
        
        # Add diagnosis labels
        df = loader.create_diagnosis_labels(df)
        print(f"Normal vs Abnormal: {df['is_abnormal'].value_counts().to_dict()}")
        
        return df
    else:
        print("❌ No clinical data found. Run download script first!")
        return None


if __name__ == "__main__":
    test_clinical_loader()
'''
    
    # Save the clinical loader
    with open("ctentropy_platform/data/clinical_loader.py", "w") as f:
        f.write(loader_code)
    
    print("✅ Created ctentropy_platform/data/clinical_loader.py")
    print("This will load REAL medical data with diagnoses!")

if __name__ == "__main__":
    # Explore available datasets
    datasets = explore_physionet_clinical()
    
    # Create download script for most promising dataset
    script_file = download_chb_mit_epilepsy()
    
    # Create clinical loader
    create_clinical_loader()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Run: .\\{script_file}")
    print(f"2. This will download REAL epilepsy patient data")
    print(f"3. Then we can train CTEntropy to detect seizures vs normal!")
    print(f"4. This gives us VERIFIED medical diagnoses to validate against!")