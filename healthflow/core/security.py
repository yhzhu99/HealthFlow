"""
Data Protection and Security Module for HealthFlow

This module implements sensitive healthcare data protection mechanisms:
- Data anonymization and pseudonymization
- Schema-only API transmission
- Mock data generation for training/testing
- Privacy-preserving data analysis
- HIPAA compliance utilities
"""

import hashlib
import json
import re
import random
import string
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path


@dataclass
class ProtectionConfig:
    """Configuration for data protection"""
    anonymization_level: str = "medium"  # low, medium, high
    preserve_structure: bool = True
    generate_mock_data: bool = False
    schema_only_mode: bool = False
    encryption_enabled: bool = False


@dataclass  
class DataClassification:
    """Classification of data sensitivity"""
    data_type: str
    sensitivity_level: str  # public, internal, confidential, restricted
    contains_pii: bool
    contains_phi: bool  # Protected Health Information
    retention_period: Optional[int] = None  # days


class DataProtector:
    """
    Comprehensive data protection system for healthcare data.
    
    Features:
    - Automatic PII/PHI detection
    - Data anonymization and pseudonymization
    - Mock data generation
    - Schema preservation
    - Audit logging
    """
    
    def __init__(self, config: Optional[ProtectionConfig] = None):
        self.config = config or ProtectionConfig()
        self.logger = logging.getLogger("DataProtector")
        
        # PII/PHI patterns
        self.pii_patterns = {
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'phone': re.compile(r'\b\d{3}-\d{3}-\d{4}\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            'medical_record_number': re.compile(r'\bMRN[\s:]?\d+\b'),
            'patient_id': re.compile(r'\b(?:patient|pt)[\s_-]?id[\s:]?\d+\b', re.IGNORECASE)
        }
        
        # Medical data patterns
        self.medical_patterns = {
            'lab_result': re.compile(r'\b\d+\.?\d*\s*(?:mg/dL|mmol/L|IU/L|ng/mL)\b'),
            'medication': re.compile(r'\b\w+\s*\d+\s*mg\b'),
            'diagnosis_code': re.compile(r'\b[A-Z]\d{2}\.\d+\b'),  # ICD-10 pattern
            'vital_signs': re.compile(r'\b(?:BP|HR|RR|Temp)[\s:]\d+\b')
        }
        
        # Audit trail
        self.protection_log: List[Dict[str, Any]] = []
    
    async def protect_data(self, data: Any) -> Dict[str, Any]:
        """
        Main data protection function that applies appropriate protection measures.
        
        Args:
            data: Raw data to protect
            
        Returns:
            Protected data with metadata about protection applied
        """
        
        if data is None:
            return {"protected_data": None, "protection_applied": False}
        
        # Classify data sensitivity
        classification = self._classify_data(data)
        
        # Apply protection based on classification
        if classification.contains_phi or classification.contains_pii:
            protected_data = await self._apply_protection(data, classification)
            protection_applied = True
        else:
            protected_data = data
            protection_applied = False
        
        # Generate schema if needed
        schema = self._generate_schema(data) if self.config.schema_only_mode else None
        
        # Generate mock data if needed
        mock_data = self._generate_mock_data(data, classification) if self.config.generate_mock_data else None
        
        # Log protection action
        self._log_protection_action(classification, protection_applied)
        
        result = {
            "protected_data": protected_data,
            "original_schema": schema,
            "mock_data": mock_data,
            "protection_applied": protection_applied,
            "classification": {
                "sensitivity_level": classification.sensitivity_level,
                "contains_pii": classification.contains_pii,
                "contains_phi": classification.contains_phi
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _classify_data(self, data: Any) -> DataClassification:
        """Classify data based on content analysis"""
        
        data_str = json.dumps(data, default=str) if not isinstance(data, str) else data
        data_str_lower = data_str.lower()
        
        # Check for PHI (Protected Health Information)
        contains_phi = any([
            pattern.search(data_str) for pattern in self.medical_patterns.values()
        ]) or any([
            keyword in data_str_lower 
            for keyword in ['patient', 'medical', 'diagnosis', 'treatment', 'medication', 'doctor', 'hospital']
        ])
        
        # Check for PII (Personally Identifiable Information)
        contains_pii = any([
            pattern.search(data_str) for pattern in self.pii_patterns.values()
        ]) or any([
            keyword in data_str_lower
            for keyword in ['name', 'address', 'birthday', 'birth', 'ssn', 'social security']
        ])
        
        # Determine sensitivity level
        if contains_phi and contains_pii:
            sensitivity_level = "restricted"
        elif contains_phi or contains_pii:
            sensitivity_level = "confidential"
        elif any(keyword in data_str_lower for keyword in ['medical', 'health', 'clinical']):
            sensitivity_level = "internal"
        else:
            sensitivity_level = "public"
        
        # Determine data type
        if isinstance(data, dict):
            data_type = "structured"
        elif isinstance(data, list):
            data_type = "list"
        elif isinstance(data, str):
            data_type = "text"
        else:
            data_type = "other"
        
        return DataClassification(
            data_type=data_type,
            sensitivity_level=sensitivity_level,
            contains_pii=contains_pii,
            contains_phi=contains_phi
        )
    
    async def _apply_protection(self, data: Any, classification: DataClassification) -> Any:
        """Apply appropriate protection measures based on classification"""
        
        if self.config.anonymization_level == "high":
            return await self._anonymize_data(data, classification)
        elif self.config.anonymization_level == "medium":
            return await self._pseudonymize_data(data, classification)
        else:
            return await self._basic_protection(data, classification)
    
    async def _anonymize_data(self, data: Any, classification: DataClassification) -> Any:
        """Apply strong anonymization (irreversible)"""
        
        if isinstance(data, dict):
            anonymized = {}
            for key, value in data.items():
                if self._is_sensitive_field(key):
                    anonymized[key] = self._anonymize_value(value, key)
                else:
                    anonymized[key] = await self._anonymize_data(value, classification)
            return anonymized
        
        elif isinstance(data, list):
            return [await self._anonymize_data(item, classification) for item in data]
        
        elif isinstance(data, str):
            return self._anonymize_text(data)
        
        else:
            return data
    
    async def _pseudonymize_data(self, data: Any, classification: DataClassification) -> Any:
        """Apply pseudonymization (reversible with key)"""
        
        if isinstance(data, dict):
            pseudonymized = {}
            for key, value in data.items():
                if self._is_sensitive_field(key):
                    pseudonymized[key] = self._pseudonymize_value(value, key)
                else:
                    pseudonymized[key] = await self._pseudonymize_data(value, classification)
            return pseudonymized
        
        elif isinstance(data, list):
            return [await self._pseudonymize_data(item, classification) for item in data]
        
        elif isinstance(data, str):
            return self._pseudonymize_text(data)
        
        else:
            return data
    
    async def _basic_protection(self, data: Any, classification: DataClassification) -> Any:
        """Apply basic protection measures"""
        
        if isinstance(data, str):
            # Just mask obvious PII patterns
            protected = data
            for pattern_name, pattern in self.pii_patterns.items():
                protected = pattern.sub(f"[{pattern_name.upper()}_REDACTED]", protected)
            return protected
        
        else:
            return data
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if a field name indicates sensitive data"""
        
        sensitive_keywords = [
            'name', 'address', 'phone', 'email', 'ssn', 'dob', 'birthday',
            'patient_id', 'mrn', 'medical_record', 'diagnosis', 'medication'
        ]
        
        field_lower = field_name.lower()
        return any(keyword in field_lower for keyword in sensitive_keywords)
    
    def _anonymize_value(self, value: Any, field_name: str) -> str:
        """Anonymize a specific value based on field type"""
        
        field_lower = field_name.lower()
        
        if 'name' in field_lower:
            return "[NAME_ANONYMIZED]"
        elif 'address' in field_lower:
            return "[ADDRESS_ANONYMIZED]"
        elif 'phone' in field_lower:
            return "[PHONE_ANONYMIZED]"
        elif 'email' in field_lower:
            return "[EMAIL_ANONYMIZED]"
        elif any(id_term in field_lower for id_term in ['id', 'ssn', 'mrn']):
            return "[ID_ANONYMIZED]"
        else:
            return "[SENSITIVE_DATA_ANONYMIZED]"
    
    def _pseudonymize_value(self, value: Any, field_name: str) -> str:
        """Pseudonymize a value (reversible with key)"""
        
        # Create consistent pseudonym based on hash
        value_str = str(value)
        hash_obj = hashlib.sha256(value_str.encode())
        hash_hex = hash_obj.hexdigest()
        
        field_lower = field_name.lower()
        
        if 'name' in field_lower:
            # Generate pseudonym name
            first_names = ['John', 'Jane', 'Alex', 'Sam', 'Chris', 'Taylor']
            last_names = ['Smith', 'Johnson', 'Brown', 'Davis', 'Wilson', 'Miller']
            idx1 = int(hash_hex[:2], 16) % len(first_names)
            idx2 = int(hash_hex[2:4], 16) % len(last_names)
            return f"{first_names[idx1]} {last_names[idx2]}"
        
        elif 'phone' in field_lower:
            # Generate pseudonym phone
            area_code = int(hash_hex[:3], 16) % 800 + 200
            exchange = int(hash_hex[3:6], 16) % 800 + 200
            number = int(hash_hex[6:10], 16) % 10000
            return f"{area_code:03d}-{exchange:03d}-{number:04d}"
        
        elif 'email' in field_lower:
            # Generate pseudonym email
            username = f"user{hash_hex[:8]}"
            domains = ['example.com', 'test.org', 'sample.net']
            domain = domains[int(hash_hex[8:10], 16) % len(domains)]
            return f"{username}@{domain}"
        
        elif any(id_term in field_lower for id_term in ['id', 'ssn', 'mrn']):
            # Generate pseudonym ID
            pseudo_id = int(hash_hex[:8], 16) % 1000000
            return f"PSEUDO_{pseudo_id:06d}"
        
        else:
            # Generic pseudonymization
            return f"PSEUDO_{hash_hex[:8].upper()}"
    
    def _anonymize_text(self, text: str) -> str:
        """Anonymize text by replacing PII patterns"""
        
        anonymized = text
        
        for pattern_name, pattern in self.pii_patterns.items():
            anonymized = pattern.sub(f"[{pattern_name.upper()}_ANONYMIZED]", anonymized)
        
        return anonymized
    
    def _pseudonymize_text(self, text: str) -> str:
        """Pseudonymize text by replacing PII patterns with consistent pseudonyms"""
        
        pseudonymized = text
        
        for pattern_name, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            for match in set(matches):  # Use set to avoid duplicate replacements
                pseudonym = self._pseudonymize_value(match, pattern_name)
                pseudonymized = pseudonymized.replace(match, pseudonym)
        
        return pseudonymized
    
    def _generate_schema(self, data: Any) -> Dict[str, Any]:
        """Generate schema of data structure without actual values"""
        
        if isinstance(data, dict):
            schema = {}
            for key, value in data.items():
                if self._is_sensitive_field(key):
                    schema[key] = {"type": "sensitive", "data_type": type(value).__name__}
                else:
                    schema[key] = self._generate_schema(value)
            return schema
        
        elif isinstance(data, list):
            if data:
                return {"type": "list", "item_schema": self._generate_schema(data[0]), "length": len(data)}
            else:
                return {"type": "list", "item_schema": None, "length": 0}
        
        else:
            return {"type": type(data).__name__, "value_present": data is not None}
    
    def _generate_mock_data(self, data: Any, classification: DataClassification) -> Any:
        """Generate mock data that preserves structure but not content"""
        
        if isinstance(data, dict):
            mock = {}
            for key, value in data.items():
                mock[key] = self._generate_mock_value(key, value)
            return mock
        
        elif isinstance(data, list):
            return [self._generate_mock_data(item, classification) for item in data[:3]]  # Limit to 3 items
        
        else:
            return self._generate_mock_value("generic", data)
    
    def _generate_mock_value(self, field_name: str, original_value: Any) -> Any:
        """Generate mock value based on field type"""
        
        field_lower = field_name.lower()
        
        if isinstance(original_value, str):
            if 'name' in field_lower:
                return random.choice(['Mock Patient', 'Test User', 'Sample Person'])
            elif 'email' in field_lower:
                return f"test.user{random.randint(1,999)}@example.com"
            elif 'phone' in field_lower:
                return f"555-{random.randint(100,999):03d}-{random.randint(1000,9999):04d}"
            elif 'address' in field_lower:
                return f"{random.randint(100,999)} Mock St, Test City, TC {random.randint(10000,99999)}"
            elif any(id_term in field_lower for id_term in ['id', 'mrn']):
                return f"MOCK_{random.randint(100000,999999)}"
            else:
                # Generic string mock
                return f"mock_{len(original_value)}_char_string"
        
        elif isinstance(original_value, (int, float)):
            if 'age' in field_lower:
                return random.randint(18, 90)
            elif any(term in field_lower for term in ['weight', 'height', 'bp', 'hr']):
                return round(random.uniform(50, 200), 1)
            else:
                return random.randint(1, 1000)
        
        elif isinstance(original_value, bool):
            return random.choice([True, False])
        
        elif isinstance(original_value, datetime):
            return datetime.now() - timedelta(days=random.randint(1, 365))
        
        else:
            return "MOCK_DATA"
    
    def _log_protection_action(self, classification: DataClassification, protection_applied: bool):
        """Log data protection actions for audit trail"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "classification": {
                "data_type": classification.data_type,
                "sensitivity_level": classification.sensitivity_level,
                "contains_pii": classification.contains_pii,
                "contains_phi": classification.contains_phi
            },
            "protection_applied": protection_applied,
            "config": {
                "anonymization_level": self.config.anonymization_level,
                "preserve_structure": self.config.preserve_structure,
                "schema_only_mode": self.config.schema_only_mode
            }
        }
        
        self.protection_log.append(log_entry)
        
        # Keep log size manageable
        if len(self.protection_log) > 1000:
            self.protection_log = self.protection_log[-500:]
    
    def get_protection_statistics(self) -> Dict[str, Any]:
        """Get statistics about data protection activities"""
        
        if not self.protection_log:
            return {"total_protections": 0, "message": "No protection activities logged"}
        
        total_protections = len(self.protection_log)
        
        # Count by sensitivity level
        sensitivity_counts = {}
        pii_count = 0
        phi_count = 0
        
        for entry in self.protection_log:
            classification = entry["classification"]
            level = classification["sensitivity_level"]
            sensitivity_counts[level] = sensitivity_counts.get(level, 0) + 1
            
            if classification["contains_pii"]:
                pii_count += 1
            if classification["contains_phi"]:
                phi_count += 1
        
        return {
            "total_protections": total_protections,
            "sensitivity_distribution": sensitivity_counts,
            "pii_protections": pii_count,
            "phi_protections": phi_count,
            "config": {
                "anonymization_level": self.config.anonymization_level,
                "preserve_structure": self.config.preserve_structure,
                "schema_only_mode": self.config.schema_only_mode
            }
        }
    
    def clear_protection_log(self):
        """Clear the protection log (for privacy/storage management)"""
        self.protection_log.clear()
        self.logger.info("Protection log cleared")
    
    def export_protection_log(self, file_path: str):
        """Export protection log for compliance reporting"""
        
        with open(file_path, 'w') as f:
            json.dump({
                "export_timestamp": datetime.now().isoformat(),
                "total_entries": len(self.protection_log),
                "protection_log": self.protection_log
            }, f, indent=2)
        
        self.logger.info(f"Protection log exported to {file_path}")
    
    def validate_data_compliance(self, data: Any) -> Dict[str, Any]:
        """Validate if data meets compliance requirements"""
        
        classification = self._classify_data(data)
        
        compliance_issues = []
        
        # Check for unprotected PII/PHI
        if classification.contains_pii and self.config.anonymization_level == "low":
            compliance_issues.append("PII detected but anonymization level is low")
        
        if classification.contains_phi and not self.config.schema_only_mode:
            compliance_issues.append("PHI detected but full data transmission enabled")
        
        # Check for retention policies
        if classification.sensitivity_level in ["confidential", "restricted"]:
            if not hasattr(self.config, 'retention_policy'):
                compliance_issues.append("No retention policy defined for sensitive data")
        
        is_compliant = len(compliance_issues) == 0
        
        return {
            "is_compliant": is_compliant,
            "classification": {
                "sensitivity_level": classification.sensitivity_level,
                "contains_pii": classification.contains_pii,
                "contains_phi": classification.contains_phi
            },
            "issues": compliance_issues,
            "recommendations": self._generate_compliance_recommendations(classification, compliance_issues)
        }
    
    def _generate_compliance_recommendations(
        self, 
        classification: DataClassification,
        issues: List[str]
    ) -> List[str]:
        """Generate compliance recommendations based on issues"""
        
        recommendations = []
        
        if any("PII" in issue for issue in issues):
            recommendations.append("Increase anonymization level to 'medium' or 'high'")
            recommendations.append("Enable schema-only mode for PII transmission")
        
        if any("PHI" in issue for issue in issues):
            recommendations.append("Enable schema-only mode for PHI")
            recommendations.append("Consider additional encryption for PHI storage")
        
        if any("retention" in issue for issue in issues):
            recommendations.append("Define and implement data retention policies")
            recommendations.append("Set up automated data purging for expired records")
        
        if classification.sensitivity_level in ["confidential", "restricted"]:
            recommendations.append("Enable audit logging for sensitive data access")
            recommendations.append("Implement access controls and user authentication")
        
        return recommendations