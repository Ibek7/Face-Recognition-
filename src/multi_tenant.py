# Multi-Tenant Architecture Support

import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

class TenantStatus(Enum):
    """Tenant status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELETED = "deleted"

class ResourceType(Enum):
    """Types of resources."""
    STORAGE = "storage"
    COMPUTE = "compute"
    MEMORY = "memory"
    API_CALLS = "api_calls"
    MODELS = "models"

@dataclass
class ResourceQuota:
    """Resource quota for tenant."""
    resource_type: ResourceType
    limit: float
    current_usage: float = 0.0
    reset_date: float = field(default_factory=time.time)
    
    def get_usage_percentage(self) -> float:
        """Get usage percentage."""
        if self.limit == 0:
            return 0.0
        return (self.current_usage / self.limit) * 100
    
    def is_exceeded(self) -> bool:
        """Check if quota is exceeded."""
        return self.current_usage > self.limit
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'resource_type': self.resource_type.value,
            'limit': self.limit,
            'current_usage': self.current_usage,
            'usage_percentage': self.get_usage_percentage(),
            'is_exceeded': self.is_exceeded()
        }

@dataclass
class Tenant:
    """Tenant entity."""
    tenant_id: str
    name: str
    owner_id: str
    created_at: float = field(default_factory=time.time)
    status: TenantStatus = TenantStatus.ACTIVE
    quotas: Dict[ResourceType, ResourceQuota] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    features_enabled: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'tenant_id': self.tenant_id,
            'name': self.name,
            'owner_id': self.owner_id,
            'status': self.status.value,
            'created_at': self.created_at,
            'quotas': {rt.value: rq.to_dict() 
                      for rt, rq in self.quotas.items()},
            'metadata': self.metadata,
            'features_enabled': self.features_enabled
        }

@dataclass
class TenantRequest:
    """Request with tenant context."""
    request_id: str
    tenant_id: str
    user_id: str
    timestamp: float = field(default_factory=time.time)
    resource_type: Optional[ResourceType] = None
    resource_amount: float = 0.0
    operation: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'tenant_id': self.tenant_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp,
            'resource_type': self.resource_type.value if self.resource_type else None,
            'resource_amount': self.resource_amount,
            'operation': self.operation
        }

class TenantManager:
    """Manage tenants."""
    
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.lock = threading.RLock()
    
    def create_tenant(self, tenant_id: str, name: str, 
                     owner_id: str) -> Tenant:
        """Create new tenant."""
        with self.lock:
            if tenant_id in self.tenants:
                raise ValueError(f"Tenant {tenant_id} already exists")
            
            tenant = Tenant(tenant_id=tenant_id, name=name, owner_id=owner_id)
            self.tenants[tenant_id] = tenant
            
            return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant."""
        with self.lock:
            return self.tenants.get(tenant_id)
    
    def update_tenant_status(self, tenant_id: str, status: TenantStatus):
        """Update tenant status."""
        with self.lock:
            tenant = self.tenants.get(tenant_id)
            if tenant:
                tenant.status = status
    
    def delete_tenant(self, tenant_id: str):
        """Delete tenant."""
        with self.lock:
            if tenant_id in self.tenants:
                self.tenants[tenant_id].status = TenantStatus.DELETED
                # Could also remove, but marking deleted is safer
    
    def list_tenants(self) -> List[Tenant]:
        """List all tenants."""
        with self.lock:
            return list(self.tenants.values())
    
    def get_active_tenants(self) -> List[Tenant]:
        """Get active tenants."""
        with self.lock:
            return [t for t in self.tenants.values() 
                   if t.status == TenantStatus.ACTIVE]

class QuotaManager:
    """Manage resource quotas."""
    
    def __init__(self):
        self.quotas: Dict[str, Dict[ResourceType, ResourceQuota]] = {}
        self.lock = threading.RLock()
    
    def set_quota(self, tenant_id: str, resource_type: ResourceType, 
                 limit: float):
        """Set quota for tenant."""
        with self.lock:
            if tenant_id not in self.quotas:
                self.quotas[tenant_id] = {}
            
            self.quotas[tenant_id][resource_type] = ResourceQuota(
                resource_type=resource_type,
                limit=limit
            )
    
    def consume_quota(self, tenant_id: str, resource_type: ResourceType,
                     amount: float) -> bool:
        """Consume quota."""
        with self.lock:
            if tenant_id not in self.quotas:
                return False
            
            quota = self.quotas[tenant_id].get(resource_type)
            if not quota:
                return False
            
            if quota.current_usage + amount > quota.limit:
                return False
            
            quota.current_usage += amount
            return True
    
    def get_quota(self, tenant_id: str, 
                 resource_type: ResourceType) -> Optional[ResourceQuota]:
        """Get quota."""
        with self.lock:
            if tenant_id in self.quotas:
                return self.quotas[tenant_id].get(resource_type)
            return None
    
    def reset_quota(self, tenant_id: str, resource_type: ResourceType):
        """Reset quota."""
        with self.lock:
            if tenant_id in self.quotas:
                quota = self.quotas[tenant_id].get(resource_type)
                if quota:
                    quota.current_usage = 0.0
                    quota.reset_date = time.time()

class TenantContext:
    """Thread-local tenant context."""
    
    def __init__(self):
        self.local = threading.local()
    
    def set_tenant_id(self, tenant_id: str):
        """Set current tenant ID."""
        self.local.tenant_id = tenant_id
    
    def get_tenant_id(self) -> Optional[str]:
        """Get current tenant ID."""
        return getattr(self.local, 'tenant_id', None)
    
    def set_user_id(self, user_id: str):
        """Set current user ID."""
        self.local.user_id = user_id
    
    def get_user_id(self) -> Optional[str]:
        """Get current user ID."""
        return getattr(self.local, 'user_id', None)

class IsolationPolicy(ABC):
    """Base class for data isolation policies."""
    
    @abstractmethod
    def is_accessible(self, tenant_id: str, resource_tenant_id: str) -> bool:
        """Check if resource is accessible to tenant."""
        pass

class FullIsolation(IsolationPolicy):
    """Complete data isolation."""
    
    def is_accessible(self, tenant_id: str, resource_tenant_id: str) -> bool:
        """Check access."""
        return tenant_id == resource_tenant_id

class SharedIsolation(IsolationPolicy):
    """Shared resources with tenant groups."""
    
    def __init__(self):
        self.tenant_groups: Dict[str, List[str]] = {}
    
    def is_accessible(self, tenant_id: str, resource_tenant_id: str) -> bool:
        """Check access."""
        if tenant_id == resource_tenant_id:
            return True
        
        for group in self.tenant_groups.values():
            if tenant_id in group and resource_tenant_id in group:
                return True
        
        return False

class TenantRequestHandler:
    """Handle requests with tenant context."""
    
    def __init__(self, tenant_manager: TenantManager,
                quota_manager: QuotaManager,
                isolation_policy: IsolationPolicy):
        self.tenant_manager = tenant_manager
        self.quota_manager = quota_manager
        self.isolation_policy = isolation_policy
        self.request_history: List[TenantRequest] = []
        self.lock = threading.RLock()
    
    def handle_request(self, request: TenantRequest) -> Dict:
        """Handle tenant request."""
        
        # Validate tenant
        tenant = self.tenant_manager.get_tenant(request.tenant_id)
        if not tenant or tenant.status != TenantStatus.ACTIVE:
            return {'success': False, 'error': 'Invalid or inactive tenant'}
        
        # Check quota if resource specified
        if request.resource_type:
            if not self.quota_manager.consume_quota(
                request.tenant_id,
                request.resource_type,
                request.resource_amount
            ):
                return {'success': False, 'error': 'Quota exceeded'}
        
        # Record request
        with self.lock:
            self.request_history.append(request)
        
        return {'success': True, 'request_id': request.request_id}
    
    def get_request_history(self, tenant_id: str = None) -> List[TenantRequest]:
        """Get request history."""
        with self.lock:
            if tenant_id:
                return [r for r in self.request_history 
                       if r.tenant_id == tenant_id]
            return self.request_history.copy()

class TenantAuditLog:
    """Audit log for tenant operations."""
    
    def __init__(self):
        self.events: List[Dict] = []
        self.lock = threading.RLock()
    
    def log_event(self, tenant_id: str, event_type: str, 
                 details: Dict = None):
        """Log event."""
        with self.lock:
            self.events.append({
                'timestamp': time.time(),
                'tenant_id': tenant_id,
                'event_type': event_type,
                'details': details or {}
            })
    
    def get_audit_trail(self, tenant_id: str) -> List[Dict]:
        """Get audit trail for tenant."""
        with self.lock:
            return [e for e in self.events if e['tenant_id'] == tenant_id]

# Example usage
if __name__ == "__main__":
    # Create managers
    tenant_mgr = TenantManager()
    quota_mgr = QuotaManager()
    policy = FullIsolation()
    request_handler = TenantRequestHandler(tenant_mgr, quota_mgr, policy)
    audit_log = TenantAuditLog()
    
    # Create tenants
    tenant1 = tenant_mgr.create_tenant("tenant_1", "Acme Corp", "owner_1")
    tenant2 = tenant_mgr.create_tenant("tenant_2", "TechCorp", "owner_2")
    
    # Set quotas
    quota_mgr.set_quota("tenant_1", ResourceType.STORAGE, 1000.0)  # 1GB
    quota_mgr.set_quota("tenant_1", ResourceType.API_CALLS, 10000.0)
    
    # Create requests
    request = TenantRequest(
        request_id="req_1",
        tenant_id="tenant_1",
        user_id="user_1",
        resource_type=ResourceType.API_CALLS,
        resource_amount=100.0,
        operation="face_recognition"
    )
    
    # Handle request
    result = request_handler.handle_request(request)
    audit_log.log_event("tenant_1", "request_processed", result)
    
    print(f"Request Result: {result}")
    print(f"\nTenant 1 Status:\n{tenant1.to_dict()}")
