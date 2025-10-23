# Advanced Security Audit & Compliance System

import hashlib
import hmac
import threading
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

class ComplianceStandard(Enum):
    """Compliance standards."""
    GDPR = "gdpr"  # EU data protection
    HIPAA = "hipaa"  # Healthcare
    PCI_DSS = "pci_dss"  # Payment card industry
    SOC2 = "soc2"  # Service organization control
    ISO27001 = "iso27001"  # Information security

class AuditLevel(Enum):
    """Audit severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    severity: AuditLevel = AuditLevel.INFO
    user_id: Optional[str] = None
    resource: Optional[str] = None
    action: str = ""
    result: str = "success"
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'severity': self.severity.value,
            'user_id': self.user_id,
            'resource': self.resource,
            'action': self.action,
            'result': self.result,
            'details': self.details
        }

class CompliancePolicy:
    """Compliance policy definition."""
    
    def __init__(self, standard: ComplianceStandard, name: str, rules: List[str]):
        self.standard = standard
        self.name = name
        self.rules = rules
        self.created_at = time.time()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'standard': self.standard.value,
            'name': self.name,
            'rules': self.rules,
            'created_at': self.created_at
        }

class AuditLog:
    """Comprehensive audit log system."""
    
    def __init__(self, retention_days: int = 365):
        self.events: List[SecurityEvent] = []
        self.retention_days = retention_days
        self.lock = threading.RLock()
    
    def log_event(self, event: SecurityEvent) -> None:
        """Log security event."""
        with self.lock:
            self.events.append(event)
            self._cleanup_old_events()
    
    def _cleanup_old_events(self) -> None:
        """Remove old events beyond retention."""
        cutoff_time = time.time() - (self.retention_days * 86400)
        self.events = [e for e in self.events if e.timestamp > cutoff_time]
    
    def query_events(self, filters: Dict[str, Any] = None) -> List[SecurityEvent]:
        """Query audit events."""
        with self.lock:
            results = self.events
            
            if not filters:
                return results
            
            # Apply filters
            if 'event_type' in filters:
                results = [e for e in results if e.event_type == filters['event_type']]
            
            if 'severity' in filters:
                results = [e for e in results if e.severity == filters['severity']]
            
            if 'user_id' in filters:
                results = [e for e in results if e.user_id == filters['user_id']]
            
            if 'time_range' in filters:
                start, end = filters['time_range']
                results = [e for e in results if start <= e.timestamp <= end]
            
            return results
    
    def get_events_json(self) -> List[Dict]:
        """Get all events as JSON."""
        with self.lock:
            return [e.to_dict() for e in self.events]

class SecurityAuditor:
    """Perform security audits."""
    
    def __init__(self):
        self.audit_log = AuditLog()
        self.compliance_policies: Dict[str, CompliancePolicy] = {}
        self.violations: List[Dict] = []
        self.lock = threading.RLock()
    
    def register_policy(self, policy: CompliancePolicy) -> None:
        """Register compliance policy."""
        with self.lock:
            self.compliance_policies[policy.standard.value] = policy
    
    def audit_data_access(self, user_id: str, resource: str, 
                         action: str, success: bool = True) -> None:
        """Audit data access."""
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            event_type="data_access",
            severity=AuditLevel.WARNING if not success else AuditLevel.INFO,
            user_id=user_id,
            resource=resource,
            action=action,
            result="failure" if not success else "success"
        )
        
        self.audit_log.log_event(event)
    
    def audit_authentication(self, user_id: str, success: bool,
                           method: str = "password") -> None:
        """Audit authentication event."""
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            event_type="authentication",
            severity=AuditLevel.CRITICAL if not success else AuditLevel.INFO,
            user_id=user_id,
            action=f"login_{method}",
            result="failure" if not success else "success",
            details={'method': method}
        )
        
        self.audit_log.log_event(event)
    
    def audit_configuration_change(self, user_id: str, config_key: str,
                                  old_value: Any, new_value: Any) -> None:
        """Audit configuration changes."""
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            event_type="config_change",
            severity=AuditLevel.CRITICAL,
            user_id=user_id,
            resource=config_key,
            action="modify",
            details={
                'old_value': str(old_value),
                'new_value': str(new_value)
            }
        )
        
        self.audit_log.log_event(event)
    
    def check_compliance(self, standard: ComplianceStandard) -> Tuple[bool, List[str]]:
        """Check compliance status."""
        with self.lock:
            policy = self.compliance_policies.get(standard.value)
            
            if not policy:
                return False, ["Policy not registered"]
            
            violations = []
            
            # Check policy rules
            for rule in policy.rules:
                if not self._validate_rule(rule):
                    violations.append(f"Rule violation: {rule}")
            
            return len(violations) == 0, violations
    
    def _validate_rule(self, rule: str) -> bool:
        """Validate compliance rule."""
        # Simplified rule validation
        if "audit_log_enabled" in rule:
            return len(self.audit_log.events) > 0
        if "encryption_enabled" in rule:
            return True  # Assume enabled
        if "access_control_enabled" in rule:
            return True  # Assume enabled
        
        return True
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        data = str(time.time()).encode()
        return hashlib.sha256(data).hexdigest()[:16]
    
    def generate_compliance_report(self, standard: ComplianceStandard) -> Dict:
        """Generate compliance report."""
        is_compliant, violations = self.check_compliance(standard)
        
        with self.lock:
            total_events = len(self.audit_log.events)
            critical_events = len([
                e for e in self.audit_log.events 
                if e.severity == AuditLevel.CRITICAL
            ])
            
            return {
                'standard': standard.value,
                'compliant': is_compliant,
                'violations': violations,
                'audit_metrics': {
                    'total_events': total_events,
                    'critical_events': critical_events,
                    'generated_at': time.time()
                }
            }

class IntegrityVerifier:
    """Verify data integrity."""
    
    def __init__(self):
        self.checksums: Dict[str, str] = {}
        self.lock = threading.RLock()
    
    def compute_checksum(self, data: Any) -> str:
        """Compute data checksum."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def store_checksum(self, key: str, data: Any) -> str:
        """Store and return checksum."""
        checksum = self.compute_checksum(data)
        
        with self.lock:
            self.checksums[key] = checksum
        
        return checksum
    
    def verify_integrity(self, key: str, data: Any) -> bool:
        """Verify data integrity."""
        with self.lock:
            if key not in self.checksums:
                return False
            
            current_checksum = self.compute_checksum(data)
            stored_checksum = self.checksums[key]
            
            return current_checksum == stored_checksum

class IncidentResponse:
    """Handle security incidents."""
    
    def __init__(self, auditor: SecurityAuditor):
        self.auditor = auditor
        self.incidents: List[Dict] = []
        self.lock = threading.RLock()
    
    def report_incident(self, incident_type: str, severity: AuditLevel,
                       description: str, evidence: Dict = None) -> str:
        """Report security incident."""
        incident_id = hashlib.sha256(
            str(time.time()).encode()
        ).hexdigest()[:16]
        
        incident = {
            'incident_id': incident_id,
            'type': incident_type,
            'severity': severity.value,
            'description': description,
            'evidence': evidence or {},
            'timestamp': time.time(),
            'status': 'open'
        }
        
        with self.lock:
            self.incidents.append(incident)
        
        return incident_id
    
    def get_open_incidents(self) -> List[Dict]:
        """Get open incidents."""
        with self.lock:
            return [i for i in self.incidents if i['status'] == 'open']
    
    def close_incident(self, incident_id: str, resolution: str) -> bool:
        """Close incident."""
        with self.lock:
            for incident in self.incidents:
                if incident['incident_id'] == incident_id:
                    incident['status'] = 'closed'
                    incident['resolution'] = resolution
                    incident['closed_at'] = time.time()
                    return True
        
        return False

# Example usage
if __name__ == "__main__":
    # Create auditor
    auditor = SecurityAuditor()
    
    # Register GDPR policy
    gdpr_policy = CompliancePolicy(
        ComplianceStandard.GDPR,
        "GDPR Compliance",
        ["audit_log_enabled", "encryption_enabled", "access_control_enabled"]
    )
    auditor.register_policy(gdpr_policy)
    
    # Log events
    auditor.audit_authentication("user123", True)
    auditor.audit_data_access("user123", "/data/faces", "read")
    auditor.audit_configuration_change("admin", "api_key", "old", "new")
    
    # Generate report
    report = auditor.generate_compliance_report(ComplianceStandard.GDPR)
    print("Compliance Report:")
    print(json.dumps(report, indent=2))
    
    # Test incident response
    incident_handler = IncidentResponse(auditor)
    incident_id = incident_handler.report_incident(
        "unauthorized_access",
        AuditLevel.CRITICAL,
        "Suspicious access detected",
        {'ip': '192.168.1.1', 'user': 'user123'}
    )
    print(f"\nIncident reported: {incident_id}")
