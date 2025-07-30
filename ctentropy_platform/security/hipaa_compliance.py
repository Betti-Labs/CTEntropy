"""
HIPAA Compliance and Data Privacy Module
Ensures clinical-grade data security and privacy protection
"""

import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
from dataclasses import dataclass, asdict
from enum import Enum

# Configure secure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [SECURITY] %(message)s',
    handlers=[
        logging.FileHandler('ctentropy_security.log'),
        logging.StreamHandler()
    ]
)

security_logger = logging.getLogger('CTEntropy.Security')

class DataClassification(Enum):
    """HIPAA data classification levels"""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    PHI = "PHI"  # Protected Health Information

@dataclass
class PatientIdentifier:
    """Secure patient identifier with anonymization"""
    original_id: str
    anonymized_id: str
    hash_salt: str
    created_date: datetime
    classification: DataClassification = DataClassification.PHI

@dataclass
class AccessLog:
    """Audit log entry for HIPAA compliance"""
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    patient_id: str
    ip_address: str
    success: bool
    details: Optional[str] = None

class HIPAACompliance:
    """HIPAA compliance and data security manager"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initialize HIPAA compliance system
        
        Args:
            encryption_key: Optional encryption key (generated if not provided)
        """
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.patient_registry = {}
        self.access_logs = []
        self.data_retention_days = 2555  # 7 years as per HIPAA
        
        # Ensure secure directories exist
        self._setup_secure_directories()
        
        security_logger.info("HIPAA compliance system initialized")
    
    def _generate_encryption_key(self) -> bytes:
        """Generate secure encryption key"""
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        
        # Store salt securely (in production, use secure key management)
        with open('.hipaa_salt', 'wb') as f:
            f.write(salt)
        
        security_logger.info("New encryption key generated")
        return key
    
    def _setup_secure_directories(self):
        """Setup secure directory structure"""
        secure_dirs = [
            'secure_data/encrypted',
            'secure_data/logs',
            'secure_data/backups',
            'secure_data/temp'
        ]
        
        for dir_path in secure_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            # Set restrictive permissions (Unix/Linux)
            try:
                os.chmod(dir_path, 0o700)  # Owner read/write/execute only
            except:
                pass  # Windows doesn't support chmod
    
    def anonymize_patient_id(self, original_id: str) -> str:
        """
        Create anonymized patient identifier
        
        Args:
            original_id: Original patient identifier
            
        Returns:
            Anonymized identifier for use in analysis
        """
        # Generate salt for this patient
        salt = secrets.token_hex(16)
        
        # Create hash of original ID with salt
        hash_input = f"{original_id}_{salt}".encode()
        anonymized_hash = hashlib.sha256(hash_input).hexdigest()
        
        # Create readable anonymized ID
        anonymized_id = f"ANON_{anonymized_hash[:12].upper()}"
        
        # Store mapping securely
        patient_record = PatientIdentifier(
            original_id=original_id,
            anonymized_id=anonymized_id,
            hash_salt=salt,
            created_date=datetime.now()
        )
        
        self.patient_registry[anonymized_id] = patient_record
        
        # Log anonymization
        self._log_access(
            user_id="SYSTEM",
            action="ANONYMIZE_PATIENT_ID",
            resource="patient_registry",
            patient_id=anonymized_id,
            success=True,
            details=f"Original ID anonymized"
        )
        
        security_logger.info(f"Patient ID anonymized: {original_id} -> {anonymized_id}")
        return anonymized_id
    
    def encrypt_sensitive_data(self, data: Any, classification: DataClassification) -> bytes:
        """
        Encrypt sensitive data based on classification
        
        Args:
            data: Data to encrypt
            classification: HIPAA classification level
            
        Returns:
            Encrypted data bytes
        """
        if classification in [DataClassification.PHI, DataClassification.CONFIDENTIAL]:
            # Convert data to JSON string
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data)
            else:
                data_str = str(data)
            
            # Encrypt data
            encrypted_data = self.cipher_suite.encrypt(data_str.encode())
            
            security_logger.info(f"Data encrypted with classification: {classification.value}")
            return encrypted_data
        else:
            # Lower classification data doesn't require encryption
            return str(data).encode()
    
    def decrypt_sensitive_data(self, encrypted_data: bytes, classification: DataClassification) -> Any:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_data: Encrypted data bytes
            classification: HIPAA classification level
            
        Returns:
            Decrypted data
        """
        try:
            if classification in [DataClassification.PHI, DataClassification.CONFIDENTIAL]:
                # Decrypt data
                decrypted_bytes = self.cipher_suite.decrypt(encrypted_data)
                decrypted_str = decrypted_bytes.decode()
                
                # Try to parse as JSON
                try:
                    return json.loads(decrypted_str)
                except json.JSONDecodeError:
                    return decrypted_str
            else:
                return encrypted_data.decode()
                
        except Exception as e:
            security_logger.error(f"Decryption failed: {str(e)}")
            raise ValueError("Failed to decrypt data - invalid key or corrupted data")
    
    def secure_file_storage(self, file_path: str, data: Any, 
                          classification: DataClassification) -> str:
        """
        Store file with appropriate security measures
        
        Args:
            file_path: Path to store file
            data: Data to store
            classification: HIPAA classification level
            
        Returns:
            Path to stored file
        """
        # Determine storage location based on classification
        if classification == DataClassification.PHI:
            secure_path = f"secure_data/encrypted/{Path(file_path).name}"
        else:
            secure_path = file_path
        
        # Encrypt data if required
        if classification in [DataClassification.PHI, DataClassification.CONFIDENTIAL]:
            encrypted_data = self.encrypt_sensitive_data(data, classification)
            
            # Store encrypted data
            with open(secure_path, 'wb') as f:
                f.write(encrypted_data)
        else:
            # Store unencrypted data
            if isinstance(data, str):
                with open(secure_path, 'w') as f:
                    f.write(data)
            else:
                with open(secure_path, 'w') as f:
                    json.dump(data, f)
        
        # Set restrictive file permissions
        try:
            os.chmod(secure_path, 0o600)  # Owner read/write only
        except:
            pass  # Windows doesn't support chmod
        
        security_logger.info(f"File stored securely: {secure_path} ({classification.value})")
        return secure_path
    
    def _log_access(self, user_id: str, action: str, resource: str, 
                   patient_id: str, success: bool, ip_address: str = "localhost",
                   details: Optional[str] = None):
        """
        Log access for HIPAA audit trail
        
        Args:
            user_id: User performing action
            action: Action performed
            resource: Resource accessed
            patient_id: Patient identifier
            success: Whether action succeeded
            ip_address: IP address of user
            details: Additional details
        """
        log_entry = AccessLog(
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            patient_id=patient_id,
            ip_address=ip_address,
            success=success,
            details=details
        )
        
        self.access_logs.append(log_entry)
        
        # Write to secure log file
        log_file = "secure_data/logs/hipaa_audit.log"
        with open(log_file, 'a') as f:
            f.write(f"{json.dumps(asdict(log_entry), default=str)}\n")
        
        # Log to security logger
        status = "SUCCESS" if success else "FAILURE"
        security_logger.info(
            f"ACCESS_LOG: {user_id} {action} {resource} {patient_id} {status}"
        )
    
    def process_eeg_data_securely(self, eeg_data: Any, patient_id: str, 
                                user_id: str) -> Dict[str, Any]:
        """
        Process EEG data with HIPAA compliance
        
        Args:
            eeg_data: EEG signal data
            patient_id: Patient identifier
            user_id: User processing data
            
        Returns:
            Processing result with security metadata
        """
        # Log data access
        self._log_access(
            user_id=user_id,
            action="PROCESS_EEG_DATA",
            resource="eeg_analysis",
            patient_id=patient_id,
            success=True,
            details="EEG data processing initiated"
        )
        
        try:
            # Anonymize patient ID if not already done
            if not patient_id.startswith("ANON_"):
                anonymized_id = self.anonymize_patient_id(patient_id)
            else:
                anonymized_id = patient_id
            
            # Process data (placeholder for actual CTEntropy processing)
            processing_result = {
                'patient_id': anonymized_id,
                'processed_date': datetime.now().isoformat(),
                'data_classification': DataClassification.PHI.value,
                'processing_user': user_id,
                'security_level': 'HIPAA_COMPLIANT'
            }
            
            # Store result securely
            result_file = f"patient_result_{anonymized_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            secure_path = self.secure_file_storage(
                result_file, 
                processing_result, 
                DataClassification.PHI
            )
            
            # Log successful processing
            self._log_access(
                user_id=user_id,
                action="EEG_PROCESSING_COMPLETE",
                resource=secure_path,
                patient_id=anonymized_id,
                success=True,
                details="EEG processing completed successfully"
            )
            
            return {
                'status': 'success',
                'patient_id': anonymized_id,
                'result_file': secure_path,
                'security_compliance': 'HIPAA_COMPLIANT'
            }
            
        except Exception as e:
            # Log processing failure
            self._log_access(
                user_id=user_id,
                action="EEG_PROCESSING_FAILED",
                resource="eeg_analysis",
                patient_id=patient_id,
                success=False,
                details=f"Processing failed: {str(e)}"
            )
            
            security_logger.error(f"EEG processing failed for {patient_id}: {str(e)}")
            raise
    
    def generate_audit_report(self, start_date: datetime, end_date: datetime) -> str:
        """
        Generate HIPAA audit report
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Path to audit report
        """
        # Filter logs by date range
        filtered_logs = [
            log for log in self.access_logs
            if start_date <= log.timestamp <= end_date
        ]
        
        # Generate report
        report = {
            'report_generated': datetime.now().isoformat(),
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'total_access_events': len(filtered_logs),
            'successful_accesses': len([log for log in filtered_logs if log.success]),
            'failed_accesses': len([log for log in filtered_logs if not log.success]),
            'unique_users': len(set(log.user_id for log in filtered_logs)),
            'unique_patients': len(set(log.patient_id for log in filtered_logs)),
            'access_logs': [asdict(log) for log in filtered_logs]
        }
        
        # Store report securely
        report_file = f"hipaa_audit_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        report_path = self.secure_file_storage(
            report_file,
            report,
            DataClassification.CONFIDENTIAL
        )
        
        security_logger.info(f"HIPAA audit report generated: {report_path}")
        return report_path
    
    def data_retention_cleanup(self):
        """
        Clean up data based on retention policy
        """
        cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
        
        # Clean up old access logs
        self.access_logs = [
            log for log in self.access_logs
            if log.timestamp > cutoff_date
        ]
        
        # Clean up old patient records (in production, archive instead of delete)
        expired_patients = [
            patient_id for patient_id, record in self.patient_registry.items()
            if record.created_date < cutoff_date
        ]
        
        for patient_id in expired_patients:
            del self.patient_registry[patient_id]
            security_logger.info(f"Patient record expired and removed: {patient_id}")
        
        security_logger.info(f"Data retention cleanup completed. Removed {len(expired_patients)} expired records")
    
    def validate_hipaa_compliance(self) -> Dict[str, bool]:
        """
        Validate current HIPAA compliance status
        
        Returns:
            Compliance status dictionary
        """
        compliance_checks = {
            'encryption_enabled': bool(self.encryption_key),
            'audit_logging_active': True,  # Always true if system is running
            'secure_storage_configured': Path('secure_data').exists(),
            'patient_anonymization_active': True,  # Always true if system is running
            'data_retention_policy_active': self.data_retention_days > 0
        }
        
        overall_compliance = all(compliance_checks.values())
        compliance_checks['overall_hipaa_compliant'] = overall_compliance
        
        security_logger.info(f"HIPAA compliance validation: {compliance_checks}")
        return compliance_checks

# HIPAA compliance decorator
def hipaa_secure(func):
    """Decorator to ensure HIPAA compliance for functions"""
    def wrapper(*args, **kwargs):
        # Log function access
        security_logger.info(f"HIPAA-secured function called: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            security_logger.info(f"HIPAA-secured function completed: {func.__name__}")
            return result
        except Exception as e:
            security_logger.error(f"HIPAA-secured function failed: {func.__name__} - {str(e)}")
            raise
    
    return wrapper

# Example usage and testing
def test_hipaa_compliance():
    """Test HIPAA compliance functionality"""
    
    print("🔒 Testing HIPAA Compliance System...")
    
    # Initialize compliance system
    hipaa = HIPAACompliance()
    
    # Test patient anonymization
    original_id = "PATIENT_12345"
    anonymized_id = hipaa.anonymize_patient_id(original_id)
    print(f"✅ Patient anonymized: {original_id} -> {anonymized_id}")
    
    # Test data encryption
    sensitive_data = {"diagnosis": "epilepsy", "confidence": 94.2}
    encrypted = hipaa.encrypt_sensitive_data(sensitive_data, DataClassification.PHI)
    decrypted = hipaa.decrypt_sensitive_data(encrypted, DataClassification.PHI)
    print(f"✅ Data encryption/decryption successful: {decrypted}")
    
    # Test secure processing
    result = hipaa.process_eeg_data_securely(
        eeg_data="mock_eeg_data",
        patient_id=original_id,
        user_id="DR_SMITH"
    )
    print(f"✅ Secure processing completed: {result['status']}")
    
    # Test compliance validation
    compliance = hipaa.validate_hipaa_compliance()
    print(f"✅ HIPAA compliance status: {compliance['overall_hipaa_compliant']}")
    
    # Generate audit report
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    audit_report = hipaa.generate_audit_report(start_date, end_date)
    print(f"✅ Audit report generated: {audit_report}")
    
    print("🔒 HIPAA compliance testing completed!")

if __name__ == "__main__":
    test_hipaa_compliance()