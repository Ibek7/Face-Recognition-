# Audit Logging & Compliance System

import threading
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import hashlib

class AuditEventType(Enum):
    """Types of audit events."""
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    DATA_ACCESS = "DATA_ACCESS"
    DATA_MODIFICATION = "DATA_MODIFICATION"
    DATA_DELETION = "DATA_DELETION"
    CONFIGURATION_CHANGE = "CONFIGURATION_CHANGE"
    SECURITY_EVENT = "SECURITY_EVENT"
    COMPLIANCE_CHECK = "COMPLIANCE_CHECK"
    ERROR = "ERROR"

class AuditSeverity(Enum):
    """Audit event severity."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

@dataclass
class AuditEvent:
    """Audit log event."""
    event_id: str
    timestamp: float
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: str
    resource: str
    action: str
    details: Dict[str, Any]
    ip_address: str = ""
    status: str = "SUCCESS"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return data
    
    def calculate_hash(self) -> str:
        """Calculate tamper-proof hash."""
        event_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()

class AuditLogger:
    """Central audit logger."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.events: List[AuditEvent] = []
        self.event_handlers: List[callable] = []
        self.lock = threading.RLock()
    
    def log_event(self, event_type: AuditEventType, user_id: str,
                  resource: str, action: str, details: Dict = None,
                  severity: AuditSeverity = AuditSeverity.INFO,
                  ip_address: str = "", status: str = "SUCCESS") -> AuditEvent:
        """Log audit event."""
        import uuid
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().timestamp(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            resource=resource,
            action=action,
            details=details or {},
            ip_address=ip_address,
            status=status
        )
        
        with self.lock:
            self.events.append(event)
            
            # Call handlers
            for handler in self.event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Error in audit handler: {e}")
        
        return event
    
    def add_event_handler(self, handler: callable) -> None:
        """Add event handler."""
        with self.lock:
            self.event_handlers.append(handler)
    
    def get_events(self, start_date: datetime = None, 
                   end_date: datetime = None,
                   user_id: str = None,
                   event_type: AuditEventType = None) -> List[AuditEvent]:
        """Get audit events with filtering."""
        with self.lock:
            results = self.events
            
            if start_date:
                results = [e for e in results if e.timestamp >= start_date.timestamp()]
            
            if end_date:
                results = [e for e in results if e.timestamp <= end_date.timestamp()]
            
            if user_id:
                results = [e for e in results if e.user_id == user_id]
            
            if event_type:
                results = [e for e in results if e.event_type == event_type]
            
            return results
    
    def export_logs(self) -> List[Dict]:
        """Export all audit logs."""
        with self.lock:
            return [event.to_dict() for event in self.events]

class ComplianceChecker:
    """Compliance policy enforcement."""
    
    def __init__(self):
        self.policies: Dict[str, callable] = {}
        self.violations: List[Dict] = []
        self.lock = threading.RLock()
    
    def add_policy(self, policy_name: str, policy_check: callable) -> None:
        """Add compliance policy."""
        with self.lock:
            self.policies[policy_name] = policy_check
    
    def check_compliance(self, audit_event: AuditEvent) -> List[str]:
        """Check event against policies."""
        violations = []
        
        with self.lock:
            for policy_name, policy_check in self.policies.items():
                try:
                    if not policy_check(audit_event):
                        violations.append(policy_name)
                        self.violations.append({
                            'event_id': audit_event.event_id,
                            'policy': policy_name,
                            'timestamp': audit_event.timestamp
                        })
                except Exception as e:
                    print(f"Error checking policy {policy_name}: {e}")
        
        return violations
    
    def get_violations(self) -> List[Dict]:
        """Get compliance violations."""
        with self.lock:
            return self.violations.copy()

class DataAccessLog:
    """Log and track data access patterns."""
    
    def __init__(self):
        self.access_records: List[Dict] = []
        self.lock = threading.RLock()
    
    def log_access(self, user_id: str, resource_id: str,
                   access_type: str, timestamp: float = None) -> None:
        """Log data access."""
        import time
        
        if timestamp is None:
            timestamp = time.time()
        
        record = {
            'user_id': user_id,
            'resource_id': resource_id,
            'access_type': access_type,
            'timestamp': timestamp
        }
        
        with self.lock:
            self.access_records.append(record)
    
    def get_access_patterns(self, user_id: str = None) -> Dict:
        """Get access patterns."""
        with self.lock:
            if user_id:
                records = [r for r in self.access_records if r['user_id'] == user_id]
            else:
                records = self.access_records
            
            # Aggregate by resource
            patterns = {}
            for record in records:
                resource = record['resource_id']
                if resource not in patterns:
                    patterns[resource] = {'count': 0, 'types': {}}
                
                patterns[resource]['count'] += 1
                access_type = record['access_type']
                patterns[resource]['types'][access_type] = \
                    patterns[resource]['types'].get(access_type, 0) + 1
            
            return patterns

class ChangeTracker:
    """Track configuration and data changes."""
    
    def __init__(self):
        self.changes: List[Dict] = []
        self.lock = threading.RLock()
    
    def record_change(self, entity_type: str, entity_id: str,
                     change_type: str, old_value: Any, new_value: Any,
                     user_id: str, reason: str = "") -> None:
        """Record configuration change."""
        change = {
            'entity_type': entity_type,
            'entity_id': entity_id,
            'change_type': change_type,
            'old_value': old_value,
            'new_value': new_value,
            'user_id': user_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        with self.lock:
            self.changes.append(change)
    
    def get_change_history(self, entity_type: str = None,
                          entity_id: str = None) -> List[Dict]:
        """Get change history."""
        with self.lock:
            results = self.changes
            
            if entity_type:
                results = [c for c in results if c['entity_type'] == entity_type]
            
            if entity_id:
                results = [c for c in results if c['entity_id'] == entity_id]
            
            return results

class ComplianceReport:
    """Generate compliance reports."""
    
    @staticmethod
    def generate_access_report(audit_logger: AuditLogger,
                              start_date: datetime,
                              end_date: datetime) -> Dict:
        """Generate data access report."""
        events = audit_logger.get_events(start_date, end_date,
                                        event_type=AuditEventType.DATA_ACCESS)
        
        users = {}
        resources = {}
        
        for event in events:
            user = event.user_id
            resource = event.resource
            
            if user not in users:
                users[user] = {'count': 0, 'resources': set()}
            users[user]['count'] += 1
            users[user]['resources'].add(resource)
            
            if resource not in resources:
                resources[resource] = {'count': 0, 'users': set()}
            resources[resource]['count'] += 1
            resources[resource]['users'].add(event.user_id)
        
        return {
            'period': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
            'total_accesses': len(events),
            'unique_users': len(users),
            'unique_resources': len(resources),
            'user_summary': {u: {'accesses': v['count'], 'resources': len(v['resources'])}
                           for u, v in users.items()},
            'resource_summary': {r: {'accesses': v['count'], 'users': len(v['users'])}
                               for r, v in resources.items()}
        }

# Example usage
if __name__ == "__main__":
    logger = AuditLogger("face-recognition-service")
    checker = ComplianceChecker()
    
    # Add compliance policy
    def no_bulk_delete(event: AuditEvent) -> bool:
        if event.event_type == AuditEventType.DATA_DELETION:
            count = event.details.get('record_count', 0)
            return count < 100  # Allow only <100 records per delete
        return True
    
    checker.add_policy("bulk_delete_prevention", no_bulk_delete)
    
    # Log events
    event = logger.log_event(
        AuditEventType.DATA_ACCESS,
        user_id="user123",
        resource="face_database",
        action="query",
        details={"query_type": "similarity_search"},
        ip_address="192.168.1.1"
    )
    
    print(f"Audit Event: {json.dumps(event.to_dict(), indent=2, default=str)}")
    
    # Check compliance
    violations = checker.check_compliance(event)
    print(f"Violations: {violations}")
