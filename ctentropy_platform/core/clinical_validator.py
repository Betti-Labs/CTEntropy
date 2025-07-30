"""
Clinical-Grade Data Validation and Error Handling
Ensures robust processing of EEG data with comprehensive error recovery
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict, Any
import warnings
from dataclasses import dataclass
from enum import Enum

# Configure clinical-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ctentropy_clinical.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Validation status codes for clinical reporting"""
    VALID = "VALID"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class ValidationResult:
    """Structured validation result for clinical traceability"""
    status: ValidationStatus
    message: str
    code: str
    recommendations: list
    processed_signal: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

class ClinicalValidator:
    """Clinical-grade EEG data validation and error handling"""
    
    def __init__(self):
        self.min_duration = 0.5  # seconds
        self.max_duration = 3600  # seconds (1 hour)
        self.min_sampling_rate = 100  # Hz
        self.max_sampling_rate = 2000  # Hz
        self.min_amplitude = 1e-9  # microvolts
        self.max_amplitude = 1000  # microvolts
        
    def validate_eeg_signal(self, signal: np.ndarray, sampling_rate: float, 
                          patient_id: str = "UNKNOWN") -> ValidationResult:
        """
        Comprehensive EEG signal validation with clinical error handling
        
        Args:
            signal: EEG signal array
            sampling_rate: Sampling frequency in Hz
            patient_id: Patient identifier for logging
            
        Returns:
            ValidationResult with status and processed signal
        """
        try:
            logger.info(f"Validating EEG signal for patient {patient_id}")
            
            # Initialize validation result
            result = ValidationResult(
                status=ValidationStatus.VALID,
                message="Signal validation passed",
                code="VAL_000",
                recommendations=[],
                metadata={
                    'patient_id': patient_id,
                    'original_length': len(signal) if signal is not None else 0,
                    'original_sampling_rate': sampling_rate
                }
            )
            
            # 1. Basic input validation
            input_validation = self._validate_input(signal, sampling_rate)
            if input_validation.status != ValidationStatus.VALID:
                return input_validation
            
            # 2. Signal quality checks
            quality_result = self._check_signal_quality(signal, sampling_rate)
            if quality_result.status == ValidationStatus.CRITICAL:
                return quality_result
            
            # 3. Artifact detection and removal
            cleaned_signal, artifact_info = self._remove_artifacts(signal, sampling_rate)
            
            # 4. Signal preprocessing
            processed_signal = self._preprocess_signal(cleaned_signal, sampling_rate)
            
            # 5. Final validation
            final_validation = self._final_quality_check(processed_signal, sampling_rate)
            
            # Compile results
            result.processed_signal = processed_signal
            result.metadata.update({
                'artifacts_removed': artifact_info,
                'final_length': len(processed_signal),
                'processing_applied': ['artifact_removal', 'preprocessing']
            })
            
            # Merge warnings from all steps
            if quality_result.status == ValidationStatus.WARNING:
                result.status = ValidationStatus.WARNING
                result.recommendations.extend(quality_result.recommendations)
            
            if final_validation.status == ValidationStatus.WARNING:
                result.status = ValidationStatus.WARNING
                result.recommendations.extend(final_validation.recommendations)
            
            logger.info(f"Signal validation completed for patient {patient_id}: {result.status}")
            return result
            
        except Exception as e:
            logger.error(f"Critical error in signal validation for patient {patient_id}: {str(e)}")
            return ValidationResult(
                status=ValidationStatus.CRITICAL,
                message=f"Critical validation error: {str(e)}",
                code="VAL_999",
                recommendations=["Contact technical support", "Review input data format"],
                metadata={'patient_id': patient_id, 'error': str(e)}
            )
    
    def _validate_input(self, signal: np.ndarray, sampling_rate: float) -> ValidationResult:
        """Validate basic input parameters"""
        
        # Check if signal exists
        if signal is None:
            return ValidationResult(
                status=ValidationStatus.CRITICAL,
                message="No signal data provided",
                code="VAL_001",
                recommendations=["Provide valid EEG signal data"]
            )
        
        # Check signal format
        if not isinstance(signal, np.ndarray):
            try:
                signal = np.array(signal)
            except:
                return ValidationResult(
                    status=ValidationStatus.CRITICAL,
                    message="Signal data cannot be converted to numpy array",
                    code="VAL_002",
                    recommendations=["Ensure signal is numeric array format"]
                )
        
        # Check signal dimensions
        if len(signal.shape) != 1:
            if len(signal.shape) == 2 and signal.shape[0] == 1:
                signal = signal.flatten()
            else:
                return ValidationResult(
                    status=ValidationStatus.CRITICAL,
                    message=f"Signal must be 1-dimensional, got shape {signal.shape}",
                    code="VAL_003",
                    recommendations=["Provide single-channel EEG signal", "Use signal[0] for multi-channel data"]
                )
        
        # Check signal length
        if len(signal) == 0:
            return ValidationResult(
                status=ValidationStatus.CRITICAL,
                message="Signal is empty",
                code="VAL_004",
                recommendations=["Provide non-empty EEG signal"]
            )
        
        # Check sampling rate
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            return ValidationResult(
                status=ValidationStatus.CRITICAL,
                message=f"Invalid sampling rate: {sampling_rate}",
                code="VAL_005",
                recommendations=["Provide positive numeric sampling rate"]
            )
        
        # Check sampling rate range
        if sampling_rate < self.min_sampling_rate or sampling_rate > self.max_sampling_rate:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"Sampling rate {sampling_rate} Hz outside typical range ({self.min_sampling_rate}-{self.max_sampling_rate} Hz)",
                code="VAL_006",
                recommendations=["Verify sampling rate is correct", "Consider resampling if needed"]
            )
        
        # Check signal duration
        duration = len(signal) / sampling_rate
        if duration < self.min_duration:
            return ValidationResult(
                status=ValidationStatus.CRITICAL,
                message=f"Signal too short: {duration:.2f}s (minimum: {self.min_duration}s)",
                code="VAL_007",
                recommendations=["Provide longer EEG recording", "Minimum 0.5 seconds required"]
            )
        
        if duration > self.max_duration:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"Signal very long: {duration:.2f}s (maximum recommended: {self.max_duration}s)",
                code="VAL_008",
                recommendations=["Consider segmenting long recordings", "Processing may be slow"]
            )
        
        return ValidationResult(
            status=ValidationStatus.VALID,
            message="Input validation passed",
            code="VAL_000",
            recommendations=[]
        )
    
    def _check_signal_quality(self, signal: np.ndarray, sampling_rate: float) -> ValidationResult:
        """Check EEG signal quality and detect issues"""
        
        recommendations = []
        status = ValidationStatus.VALID
        
        # Check for NaN or infinite values
        if np.any(np.isnan(signal)):
            nan_count = np.sum(np.isnan(signal))
            if nan_count > len(signal) * 0.1:  # More than 10% NaN
                return ValidationResult(
                    status=ValidationStatus.CRITICAL,
                    message=f"Too many NaN values: {nan_count}/{len(signal)} ({nan_count/len(signal)*100:.1f}%)",
                    code="VAL_010",
                    recommendations=["Check EEG recording quality", "Review data acquisition process"]
                )
            else:
                recommendations.append("NaN values detected and will be interpolated")
                status = ValidationStatus.WARNING
        
        if np.any(np.isinf(signal)):
            return ValidationResult(
                status=ValidationStatus.CRITICAL,
                message="Infinite values detected in signal",
                code="VAL_011",
                recommendations=["Check EEG amplifier settings", "Review data preprocessing"]
            )
        
        # Check signal amplitude
        signal_std = np.std(signal)
        signal_max = np.max(np.abs(signal))
        
        if signal_std < self.min_amplitude:
            return ValidationResult(
                status=ValidationStatus.CRITICAL,
                message=f"Signal amplitude too low: {signal_std:.2e} µV",
                code="VAL_012",
                recommendations=["Check EEG electrode contact", "Verify amplifier gain settings"]
            )
        
        if signal_max > self.max_amplitude:
            recommendations.append(f"High amplitude detected: {signal_max:.1f} µV (possible artifacts)")
            status = ValidationStatus.WARNING
        
        # Check for flat signal (electrode disconnection)
        flat_threshold = signal_std * 0.01
        flat_samples = np.sum(np.abs(np.diff(signal)) < flat_threshold)
        if flat_samples > len(signal) * 0.5:
            return ValidationResult(
                status=ValidationStatus.CRITICAL,
                message=f"Signal appears flat: {flat_samples}/{len(signal)} samples unchanged",
                code="VAL_013",
                recommendations=["Check electrode connections", "Verify EEG amplifier function"]
            )
        
        # Check for clipping (ADC saturation)
        signal_range = np.max(signal) - np.min(signal)
        if signal_range < signal_std * 2:
            recommendations.append("Possible signal clipping detected")
            status = ValidationStatus.WARNING
        
        # Check frequency content
        try:
            fft_signal = np.fft.fft(signal)
            power_spectrum = np.abs(fft_signal[:len(signal)//2])
            freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)[:len(signal)//2]
            
            # Check for 50/60 Hz line noise
            line_freq_50 = np.argmin(np.abs(freqs - 50))
            line_freq_60 = np.argmin(np.abs(freqs - 60))
            
            if power_spectrum[line_freq_50] > np.mean(power_spectrum) * 10:
                recommendations.append("Strong 50 Hz line noise detected")
                status = ValidationStatus.WARNING
            
            if power_spectrum[line_freq_60] > np.mean(power_spectrum) * 10:
                recommendations.append("Strong 60 Hz line noise detected")
                status = ValidationStatus.WARNING
                
        except Exception as e:
            logger.warning(f"Could not analyze frequency content: {e}")
        
        return ValidationResult(
            status=status,
            message="Signal quality check completed",
            code="VAL_020",
            recommendations=recommendations
        )
    
    def _remove_artifacts(self, signal: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, Dict]:
        """Remove common EEG artifacts"""
        
        cleaned_signal = signal.copy()
        artifact_info = {'removed': [], 'interpolated_samples': 0}
        
        # Remove NaN values by interpolation
        if np.any(np.isnan(cleaned_signal)):
            nan_mask = np.isnan(cleaned_signal)
            cleaned_signal[nan_mask] = np.interp(
                np.where(nan_mask)[0],
                np.where(~nan_mask)[0],
                cleaned_signal[~nan_mask]
            )
            artifact_info['interpolated_samples'] = np.sum(nan_mask)
            artifact_info['removed'].append('NaN_interpolation')
        
        # Remove extreme outliers (>5 standard deviations)
        signal_std = np.std(cleaned_signal)
        signal_mean = np.mean(cleaned_signal)
        outlier_threshold = 5 * signal_std
        
        outlier_mask = np.abs(cleaned_signal - signal_mean) > outlier_threshold
        if np.any(outlier_mask):
            cleaned_signal[outlier_mask] = np.clip(
                cleaned_signal[outlier_mask],
                signal_mean - outlier_threshold,
                signal_mean + outlier_threshold
            )
            artifact_info['removed'].append('extreme_outliers')
        
        # Simple high-pass filter to remove DC drift
        if len(cleaned_signal) > 100:
            from scipy import signal as scipy_signal
            try:
                # 0.5 Hz high-pass filter
                nyquist = sampling_rate / 2
                high_cutoff = 0.5 / nyquist
                b, a = scipy_signal.butter(2, high_cutoff, btype='high')
                cleaned_signal = scipy_signal.filtfilt(b, a, cleaned_signal)
                artifact_info['removed'].append('DC_drift')
            except:
                logger.warning("Could not apply high-pass filter")
        
        return cleaned_signal, artifact_info
    
    def _preprocess_signal(self, signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply standard EEG preprocessing"""
        
        processed_signal = signal.copy()
        
        # Normalize signal to zero mean, unit variance
        processed_signal = (processed_signal - np.mean(processed_signal)) / np.std(processed_signal)
        
        return processed_signal
    
    def _final_quality_check(self, signal: np.ndarray, sampling_rate: float) -> ValidationResult:
        """Final validation of processed signal"""
        
        # Check if processing preserved signal characteristics
        if np.std(signal) < 0.1:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message="Signal variance very low after processing",
                code="VAL_030",
                recommendations=["Review preprocessing steps", "Check original signal quality"]
            )
        
        # Check for processing artifacts
        if np.any(np.abs(signal) > 10):
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message="High amplitude values after normalization",
                code="VAL_031",
                recommendations=["Review artifact removal", "Check for remaining outliers"]
            )
        
        return ValidationResult(
            status=ValidationStatus.VALID,
            message="Final quality check passed",
            code="VAL_000",
            recommendations=[]
        )

# Clinical error handling decorator
def clinical_error_handler(func):
    """Decorator for clinical-grade error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Clinical error in {func.__name__}: {str(e)}")
            return ValidationResult(
                status=ValidationStatus.CRITICAL,
                message=f"Processing error in {func.__name__}: {str(e)}",
                code="ERR_999",
                recommendations=["Contact technical support", "Review input data"]
            )
    return wrapper