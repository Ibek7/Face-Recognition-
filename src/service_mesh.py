# Service Mesh and Load Balancing System

import time
import random
import threading
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
import heapq

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    WEIGHTED = "weighted"
    IP_HASH = "ip_hash"

class ServiceStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ServiceInstance:
    """Service instance metadata."""
    instance_id: str
    host: str
    port: int
    weight: float = 1.0
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: float = field(default_factory=time.time)
    response_time_ms: float = 0.0
    error_count: int = 0
    success_count: int = 0
    active_connections: int = 0
    
    def get_load(self) -> float:
        """Calculate instance load."""
        return self.active_connections + (self.error_count * 10)
    
    def get_success_rate(self) -> float:
        """Get success rate."""
        total = self.success_count + self.error_count
        return (self.success_count / total * 100) if total > 0 else 0.0

class LoadBalancer(ABC):
    """Base class for load balancing."""
    
    def __init__(self, instances: List[ServiceInstance]):
        self.instances = instances
        self.lock = threading.RLock()
    
    @abstractmethod
    def select_instance(self, client_ip: str = None) -> ServiceInstance:
        """Select instance for request."""
        pass
    
    def get_healthy_instances(self) -> List[ServiceInstance]:
        """Get healthy instances."""
        return [inst for inst in self.instances 
                if inst.status == ServiceStatus.HEALTHY]

class RoundRobinBalancer(LoadBalancer):
    """Round robin load balancing."""
    
    def __init__(self, instances: List[ServiceInstance]):
        super().__init__(instances)
        self.current_index = 0
    
    def select_instance(self, client_ip: str = None) -> ServiceInstance:
        """Select instance in round robin fashion."""
        
        with self.lock:
            healthy = self.get_healthy_instances()
            if not healthy:
                healthy = self.instances
            
            instance = healthy[self.current_index % len(healthy)]
            self.current_index += 1
            
            return instance

class LeastConnectionsBalancer(LoadBalancer):
    """Least connections load balancing."""
    
    def select_instance(self, client_ip: str = None) -> ServiceInstance:
        """Select instance with least connections."""
        
        with self.lock:
            healthy = self.get_healthy_instances()
            if not healthy:
                healthy = self.instances
            
            return min(healthy, key=lambda inst: inst.active_connections)

class WeightedBalancer(LoadBalancer):
    """Weighted load balancing."""
    
    def select_instance(self, client_ip: str = None) -> ServiceInstance:
        """Select instance using weighted distribution."""
        
        with self.lock:
            healthy = self.get_healthy_instances()
            if not healthy:
                healthy = self.instances
            
            # Weighted random selection
            total_weight = sum(inst.weight for inst in healthy)
            choice = random.uniform(0, total_weight)
            
            current = 0
            for instance in healthy:
                current += instance.weight
                if choice <= current:
                    return instance
            
            return healthy[-1]

class IPHashBalancer(LoadBalancer):
    """IP hash based load balancing (session affinity)."""
    
    def select_instance(self, client_ip: str = None) -> ServiceInstance:
        """Select instance based on client IP."""
        
        with self.lock:
            healthy = self.get_healthy_instances()
            if not healthy:
                healthy = self.instances
            
            if not client_ip:
                return healthy[0]
            
            # Hash client IP to instance
            hash_value = hash(client_ip) % len(healthy)
            return healthy[hash_value]

class ServiceRegistry:
    """Registry for service instances."""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.lock = threading.RLock()
    
    def register_service(self, service_name: str, instance: ServiceInstance):
        """Register service instance."""
        with self.lock:
            if service_name not in self.services:
                self.services[service_name] = []
            self.services[service_name].append(instance)
    
    def deregister_service(self, service_name: str, instance_id: str):
        """Deregister service instance."""
        with self.lock:
            if service_name in self.services:
                self.services[service_name] = [
                    inst for inst in self.services[service_name]
                    if inst.instance_id != instance_id
                ]
    
    def get_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get instances for service."""
        with self.lock:
            return self.services.get(service_name, []).copy()
    
    def list_services(self) -> Dict[str, List[Dict]]:
        """List all services."""
        with self.lock:
            return {
                service: [
                    {
                        'instance_id': inst.instance_id,
                        'host': inst.host,
                        'port': inst.port,
                        'status': inst.status.value,
                        'success_rate': inst.get_success_rate()
                    }
                    for inst in instances
                ]
                for service, instances in self.services.items()
            }

class HealthChecker:
    """Health checking for service instances."""
    
    def __init__(self, check_interval: int = 10):
        self.check_interval = check_interval
        self.is_running = False
        self.check_thread = None
    
    def check_instance_health(self, instance: ServiceInstance) -> ServiceStatus:
        """Check health of instance."""
        
        try:
            # Simulate health check (in practice, would be HTTP request)
            # Check response time and success rate
            
            if instance.success_count == 0:
                return ServiceStatus.UNKNOWN
            
            success_rate = instance.get_success_rate()
            
            if success_rate >= 90:
                return ServiceStatus.HEALTHY
            elif success_rate >= 70:
                return ServiceStatus.DEGRADED
            else:
                return ServiceStatus.UNHEALTHY
        
        except Exception:
            return ServiceStatus.UNHEALTHY
    
    def start_health_checking(self, services: Dict[str, List[ServiceInstance]]):
        """Start periodic health checking."""
        
        if self.is_running:
            return
        
        self.is_running = True
        self.check_thread = threading.Thread(
            target=self._health_check_loop,
            args=(services,),
            daemon=True
        )
        self.check_thread.start()
    
    def _health_check_loop(self, services: Dict[str, List[ServiceInstance]]):
        """Health check loop."""
        
        while self.is_running:
            for service_name, instances in services.items():
                for instance in instances:
                    status = self.check_instance_health(instance)
                    instance.status = status
                    instance.last_health_check = time.time()
            
            time.sleep(self.check_interval)

class ServiceMesh:
    """Service mesh coordinator."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.registry = ServiceRegistry()
        self.balancers: Dict[str, LoadBalancer] = {}
        self.strategy = strategy
        self.health_checker = HealthChecker()
        self.lock = threading.RLock()
        self.request_history: List[Dict] = []
    
    def register_service(self, service_name: str, instance: ServiceInstance):
        """Register service."""
        self.registry.register_service(service_name, instance)
        self._create_balancer(service_name)
    
    def _create_balancer(self, service_name: str):
        """Create load balancer for service."""
        
        instances = self.registry.get_service_instances(service_name)
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            self.balancers[service_name] = RoundRobinBalancer(instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            self.balancers[service_name] = LeastConnectionsBalancer(instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            self.balancers[service_name] = WeightedBalancer(instances)
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            self.balancers[service_name] = IPHashBalancer(instances)
    
    def route_request(self, service_name: str, client_ip: str = None) -> Optional[ServiceInstance]:
        """Route request to appropriate instance."""
        
        balancer = self.balancers.get(service_name)
        if not balancer:
            return None
        
        instance = balancer.select_instance(client_ip)
        
        if instance:
            instance.active_connections += 1
            
            with self.lock:
                self.request_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'service': service_name,
                    'instance': instance.instance_id,
                    'client_ip': client_ip
                })
        
        return instance
    
    def release_request(self, instance: ServiceInstance, success: bool = True, 
                       response_time_ms: float = 0.0):
        """Release request from instance."""
        
        instance.active_connections = max(0, instance.active_connections - 1)
        
        if success:
            instance.success_count += 1
        else:
            instance.error_count += 1
        
        if response_time_ms > 0:
            # Update average response time
            total = instance.success_count + instance.error_count
            instance.response_time_ms = (
                (instance.response_time_ms * (total - 1) + response_time_ms) / total
            )
    
    def get_mesh_status(self) -> Dict[str, Any]:
        """Get mesh status."""
        
        with self.lock:
            return {
                'strategy': self.strategy.value,
                'services': self.registry.list_services(),
                'total_requests': len(self.request_history)
            }

# Example usage
if __name__ == "__main__":
    # Create mesh
    mesh = ServiceMesh(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
    
    # Register services
    instances = [
        ServiceInstance("inst1", "localhost", 8001, weight=1.0, status=ServiceStatus.HEALTHY),
        ServiceInstance("inst2", "localhost", 8002, weight=1.0, status=ServiceStatus.HEALTHY),
        ServiceInstance("inst3", "localhost", 8003, weight=1.0, status=ServiceStatus.HEALTHY),
    ]
    
    for inst in instances:
        mesh.register_service("face-recognition-api", inst)
    
    # Route requests
    for i in range(10):
        instance = mesh.route_request("face-recognition-api", client_ip="192.168.1.100")
        
        if instance:
            print(f"Request {i+1} routed to {instance.instance_id}")
            # Simulate processing
            mesh.release_request(instance, success=True, response_time_ms=45.2)
    
    # Get status
    status = mesh.get_mesh_status()
    print(f"\nService Mesh Status:")
    import json
    print(json.dumps(status, indent=2))
