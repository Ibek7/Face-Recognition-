# Advanced Resource Management

import threading
import time
import json
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

class ResourceType(Enum):
    """Types of resources."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    CUSTOM = "custom"

class AllocationStrategy(Enum):
    """Resource allocation strategies."""
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    FAIR_SHARE = "fair_share"

@dataclass
class ResourceQuota:
    """Resource quota definition."""
    resource_type: ResourceType
    total_available: float
    allocated: float = 0.0
    reserved: float = 0.0
    
    def available(self) -> float:
        """Get available resources."""
        return self.total_available - self.allocated - self.reserved
    
    def utilization(self) -> float:
        """Get utilization percentage."""
        return (self.allocated / self.total_available * 100) if self.total_available > 0 else 0

@dataclass
class ResourceAllocation:
    """Resource allocation record."""
    allocation_id: str
    resource_type: ResourceType
    requested: float
    allocated: float
    requester_id: str
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    deadline: Optional[float] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'allocation_id': self.allocation_id,
            'type': self.resource_type.value,
            'allocated': self.allocated,
            'requester': self.requester_id,
            'priority': self.priority,
            'active': self.is_active
        }

class ResourcePool:
    """Manage pool of resources."""
    
    def __init__(self, pool_id: str):
        self.pool_id = pool_id
        self.quotas: Dict[ResourceType, ResourceQuota] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.requester_allocations: Dict[str, List[str]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def set_quota(self, resource_type: ResourceType, total: float) -> None:
        """Set resource quota."""
        with self.lock:
            self.quotas[resource_type] = ResourceQuota(
                resource_type=resource_type,
                total_available=total
            )
    
    def allocate(self, resource_type: ResourceType, amount: float,
                requester_id: str, priority: int = 0,
                deadline: Optional[float] = None,
                strategy: AllocationStrategy = AllocationStrategy.FIRST_FIT) -> Optional[str]:
        """Allocate resources."""
        with self.lock:
            if resource_type not in self.quotas:
                return None
            
            quota = self.quotas[resource_type]
            
            if quota.available() < amount:
                return None
            
            # Create allocation
            allocation_id = f"alloc_{len(self.allocations)}"
            allocation = ResourceAllocation(
                allocation_id=allocation_id,
                resource_type=resource_type,
                requested=amount,
                allocated=amount,
                requester_id=requester_id,
                priority=priority,
                deadline=deadline
            )
            
            # Update quota
            quota.allocated += amount
            
            # Track allocation
            self.allocations[allocation_id] = allocation
            self.requester_allocations[requester_id].append(allocation_id)
            
            return allocation_id
    
    def deallocate(self, allocation_id: str) -> bool:
        """Deallocate resources."""
        with self.lock:
            if allocation_id not in self.allocations:
                return False
            
            allocation = self.allocations[allocation_id]
            resource_type = allocation.resource_type
            
            # Update quota
            self.quotas[resource_type].allocated -= allocation.allocated
            
            # Mark as inactive
            allocation.is_active = False
            
            return True
    
    def get_available(self, resource_type: ResourceType) -> float:
        """Get available resources."""
        with self.lock:
            if resource_type not in self.quotas:
                return 0.0
            
            return self.quotas[resource_type].available()
    
    def get_usage(self) -> Dict:
        """Get resource usage."""
        with self.lock:
            usage = {}
            
            for resource_type, quota in self.quotas.items():
                usage[resource_type.value] = {
                    'total': quota.total_available,
                    'allocated': quota.allocated,
                    'reserved': quota.reserved,
                    'available': quota.available(),
                    'utilization': quota.utilization()
                }
            
            return usage

class QuotaManager:
    """Manage resource quotas across projects."""
    
    def __init__(self):
        self.project_quotas: Dict[str, Dict[ResourceType, float]] = {}
        self.project_usage: Dict[str, Dict[ResourceType, float]] = defaultdict(lambda: defaultdict(float))
        self.lock = threading.RLock()
    
    def set_project_quota(self, project_id: str, resource_type: ResourceType,
                         limit: float) -> None:
        """Set project resource quota."""
        with self.lock:
            if project_id not in self.project_quotas:
                self.project_quotas[project_id] = {}
            
            self.project_quotas[project_id][resource_type] = limit
    
    def request_quota(self, project_id: str, resource_type: ResourceType,
                     amount: float) -> bool:
        """Request quota allocation."""
        with self.lock:
            if project_id not in self.project_quotas:
                return False
            
            quota = self.project_quotas[project_id].get(resource_type, 0)
            usage = self.project_usage[project_id][resource_type]
            
            if usage + amount > quota:
                return False
            
            self.project_usage[project_id][resource_type] += amount
            return True
    
    def release_quota(self, project_id: str, resource_type: ResourceType,
                     amount: float) -> None:
        """Release quota."""
        with self.lock:
            self.project_usage[project_id][resource_type] -= amount
            self.project_usage[project_id][resource_type] = max(0, self.project_usage[project_id][resource_type])
    
    def get_project_status(self, project_id: str) -> Dict:
        """Get project resource status."""
        with self.lock:
            if project_id not in self.project_quotas:
                return {}
            
            status = {}
            
            for resource_type, limit in self.project_quotas[project_id].items():
                usage = self.project_usage[project_id].get(resource_type, 0)
                
                status[resource_type.value] = {
                    'limit': limit,
                    'usage': usage,
                    'available': limit - usage,
                    'utilization': (usage / limit * 100) if limit > 0 else 0
                }
            
            return status

class ResourceScheduler:
    """Schedule resources optimally."""
    
    def __init__(self):
        self.reservations: Dict[str, List[ResourceAllocation]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def reserve_resource(self, resource_type: ResourceType, amount: float,
                        requester_id: str, start_time: float,
                        duration_sec: float) -> Optional[str]:
        """Reserve resource for future use."""
        reservation_id = f"res_{len(self.reservations)}"
        
        allocation = ResourceAllocation(
            allocation_id=reservation_id,
            resource_type=resource_type,
            requested=amount,
            allocated=amount,
            requester_id=requester_id,
            timestamp=start_time,
            deadline=start_time + duration_sec
        )
        
        with self.lock:
            # Check for conflicts
            time_key = f"{resource_type.value}_{int(start_time)}"
            
            for existing in self.reservations[time_key]:
                if self._conflicts(allocation, existing):
                    return None
            
            self.reservations[time_key].append(allocation)
        
        return reservation_id
    
    def _conflicts(self, alloc1: ResourceAllocation, alloc2: ResourceAllocation) -> bool:
        """Check if allocations conflict."""
        if alloc1.resource_type != alloc2.resource_type:
            return False
        
        if not alloc1.deadline or not alloc2.deadline:
            return False
        
        # Check time overlap
        return (alloc1.timestamp < alloc2.deadline and 
                alloc2.timestamp < alloc1.deadline)
    
    def get_optimal_schedule(self, requests: List[Tuple[ResourceType, float, float]]) -> List[float]:
        """Get optimal schedule for requests."""
        # Simplified scheduling - return start times
        start_times = []
        current_time = time.time()
        
        for resource_type, amount, duration in requests:
            start_times.append(current_time)
            current_time += duration
        
        return start_times

class ResourceMonitor:
    """Monitor resource consumption."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.alerts: List[Dict] = []
        self.lock = threading.RLock()
    
    def record_metric(self, resource_id: str, value: float) -> None:
        """Record resource metric."""
        with self.lock:
            self.metrics[resource_id].append((time.time(), value))
            
            # Keep only last 1000 points
            if len(self.metrics[resource_id]) > 1000:
                self.metrics[resource_id].pop(0)
            
            # Check for alerts
            if value > 80.0:  # 80% utilization
                self._create_alert(resource_id, 'HIGH_UTILIZATION', value)
    
    def _create_alert(self, resource_id: str, alert_type: str, value: float) -> None:
        """Create alert."""
        alert = {
            'timestamp': time.time(),
            'resource_id': resource_id,
            'type': alert_type,
            'value': value
        }
        
        self.alerts.append(alert)
    
    def get_metrics(self, resource_id: str) -> List[Tuple[float, float]]:
        """Get metrics for resource."""
        with self.lock:
            return list(self.metrics.get(resource_id, []))
    
    def get_alerts(self, resource_id: str = None) -> List[Dict]:
        """Get alerts."""
        with self.lock:
            if resource_id:
                return [a for a in self.alerts if a['resource_id'] == resource_id]
            
            return list(self.alerts)

# Example usage
if __name__ == "__main__":
    # Create resource pool
    pool = ResourcePool("pool1")
    pool.set_quota(ResourceType.CPU, 100.0)
    pool.set_quota(ResourceType.MEMORY, 1000.0)
    
    # Allocate resources
    cpu_alloc = pool.allocate(ResourceType.CPU, 20.0, "app1")
    mem_alloc = pool.allocate(ResourceType.MEMORY, 256.0, "app1")
    
    print(f"Allocations: CPU={cpu_alloc}, Memory={mem_alloc}")
    
    # Get usage
    usage = pool.get_usage()
    print(f"\nResource Usage:")
    print(json.dumps(usage, indent=2, default=str))
    
    # Quota manager
    quota_mgr = QuotaManager()
    quota_mgr.set_project_quota("proj1", ResourceType.CPU, 50.0)
    quota_mgr.set_project_quota("proj1", ResourceType.MEMORY, 500.0)
    
    can_allocate = quota_mgr.request_quota("proj1", ResourceType.CPU, 30.0)
    print(f"\nCan allocate 30 CPU: {can_allocate}")
    
    status = quota_mgr.get_project_status("proj1")
    print(f"Project Status:")
    print(json.dumps(status, indent=2, default=str))
