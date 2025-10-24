# Service Mesh Framework - Advanced

import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    RANDOM = "random"
    IP_HASH = "ip_hash"

@dataclass
class Service:
    """Service definition."""
    name: str
    namespace: str
    instances: List[Dict] = field(default_factory=list)
    version: str = "v1"
    replicas: int = 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'namespace': self.namespace,
            'version': self.version,
            'replicas': self.replicas,
            'instances': self.instances
        }

@dataclass
class ServiceInstance:
    """Service instance."""
    instance_id: str
    host: str
    port: int
    weight: float = 1.0
    healthy: bool = True
    connections: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'instance_id': self.instance_id,
            'host': self.host,
            'port': self.port,
            'weight': self.weight,
            'healthy': self.healthy,
            'connections': self.connections
        }

class LoadBalancer:
    """Load balancer for service instances."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.current_index = 0
        self.lock = threading.RLock()
    
    def select_instance(self, instances: List[ServiceInstance],
                       client_ip: str = None) -> Optional[ServiceInstance]:
        """Select instance based on strategy."""
        healthy_instances = [i for i in instances if i.healthy]
        
        if not healthy_instances:
            return None
        
        with self.lock:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                instance = healthy_instances[self.current_index % len(healthy_instances)]
                self.current_index += 1
                return instance
            
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return min(healthy_instances, key=lambda i: i.connections)
            
            elif self.strategy == LoadBalancingStrategy.WEIGHTED:
                total_weight = sum(i.weight for i in healthy_instances)
                import random
                r = random.uniform(0, total_weight)
                current = 0
                for instance in healthy_instances:
                    current += instance.weight
                    if r <= current:
                        return instance
                return healthy_instances[0]
            
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                import random
                return random.choice(healthy_instances)
            
            elif self.strategy == LoadBalancingStrategy.IP_HASH:
                if client_ip:
                    idx = hash(client_ip) % len(healthy_instances)
                    return healthy_instances[idx]
                return healthy_instances[0]
        
        return None

class ServiceRegistry:
    """Service registry for service discovery."""
    
    def __init__(self):
        self.services: Dict[str, Service] = {}
        self.instances: Dict[str, ServiceInstance] = {}
        self.lock = threading.RLock()
    
    def register_service(self, service: Service) -> None:
        """Register service."""
        with self.lock:
            self.services[service.name] = service
    
    def register_instance(self, instance: ServiceInstance, service_name: str) -> None:
        """Register service instance."""
        with self.lock:
            if service_name in self.services:
                self.services[service_name].instances.append(instance.to_dict())
            
            self.instances[instance.instance_id] = instance
    
    def deregister_instance(self, instance_id: str) -> None:
        """Deregister instance."""
        with self.lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
    
    def get_service(self, service_name: str) -> Optional[Service]:
        """Get service."""
        with self.lock:
            return self.services.get(service_name)
    
    def get_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get service instances."""
        with self.lock:
            service = self.services.get(service_name)
            if not service:
                return []
            
            return [self.instances[iid] for iid in service.instances
                   if iid in self.instances]
    
    def update_instance_health(self, instance_id: str, healthy: bool) -> None:
        """Update instance health status."""
        with self.lock:
            if instance_id in self.instances:
                self.instances[instance_id].healthy = healthy

class TrafficPolicy:
    """Traffic management policy."""
    
    def __init__(self, name: str):
        self.name = name
        self.rules: List[Dict] = []
        self.retries = 0
        self.timeout_ms = 30000
        self.circuit_breaker_threshold = 50
    
    def add_retry_policy(self, max_retries: int, backoff: str = "exponential") -> None:
        """Add retry policy."""
        self.retries = max_retries
    
    def add_timeout(self, timeout_ms: int) -> None:
        """Add timeout."""
        self.timeout_ms = timeout_ms
    
    def add_circuit_breaker(self, threshold: float) -> None:
        """Add circuit breaker."""
        self.circuit_breaker_threshold = threshold
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'retries': self.retries,
            'timeout_ms': self.timeout_ms,
            'circuit_breaker_threshold': self.circuit_breaker_threshold
        }

class ServiceMeshV2:
    """Service mesh for inter-service communication."""
    
    def __init__(self):
        self.registry = ServiceRegistry()
        self.load_balancers: Dict[str, LoadBalancer] = {}
        self.traffic_policies: Dict[str, TrafficPolicy] = {}
        self.metrics = {}
        self.lock = threading.RLock()
    
    def register_service(self, service: Service,
                        lb_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN) -> None:
        """Register service in mesh."""
        with self.lock:
            self.registry.register_service(service)
            self.load_balancers[service.name] = LoadBalancer(lb_strategy)
            self.metrics[service.name] = {
                'requests': 0,
                'errors': 0,
                'latency': []
            }
    
    def add_instance(self, service_name: str, instance: ServiceInstance) -> None:
        """Add service instance."""
        with self.lock:
            self.registry.register_instance(instance, service_name)
    
    def route_request(self, service_name: str, request: Dict,
                     client_ip: str = None) -> Optional[ServiceInstance]:
        """Route request to instance."""
        with self.lock:
            instances = self.registry.get_instances(service_name)
            lb = self.load_balancers.get(service_name)
            
            if not lb or not instances:
                return None
            
            instance = lb.select_instance(instances, client_ip)
            
            if instance:
                instance.connections += 1
                self.metrics[service_name]['requests'] += 1
            
            return instance
    
    def mark_instance_unhealthy(self, service_name: str,
                               instance_id: str) -> None:
        """Mark instance as unhealthy."""
        with self.lock:
            self.registry.update_instance_health(instance_id, False)
            self.metrics[service_name]['errors'] += 1
    
    def add_traffic_policy(self, service_name: str,
                          policy: TrafficPolicy) -> None:
        """Add traffic policy."""
        with self.lock:
            self.traffic_policies[service_name] = policy
    
    def get_mesh_status(self) -> Dict:
        """Get mesh status."""
        with self.lock:
            return {
                'services': len(self.registry.services),
                'instances': len(self.registry.instances),
                'metrics': self.metrics
            }

class CanaryDeployment:
    """Canary deployment for gradual rollouts."""
    
    def __init__(self, service_name: str, new_version: str):
        self.service_name = service_name
        self.new_version = new_version
        self.canary_percentage = 10
        self.stable_percentage = 90
        self.status = "initial"
        self.metrics = {'errors': 0, 'requests': 0}
    
    def increase_traffic(self, percentage: int) -> None:
        """Increase traffic to new version."""
        self.canary_percentage = percentage
        self.stable_percentage = 100 - percentage
    
    def should_rollback(self) -> bool:
        """Check if should rollback."""
        if self.metrics['requests'] == 0:
            return False
        
        error_rate = self.metrics['errors'] / self.metrics['requests']
        return error_rate > 0.05  # Rollback if error rate > 5%
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'service': self.service_name,
            'new_version': self.new_version,
            'canary_percentage': self.canary_percentage,
            'stable_percentage': self.stable_percentage,
            'status': self.status,
            'metrics': self.metrics
        }

class ServiceProxy:
    """Service proxy for request interception."""
    
    def __init__(self, mesh: ServiceMeshV2):
        self.mesh = mesh
        self.interceptors: List[Callable] = []
    
    def add_interceptor(self, interceptor: Callable) -> None:
        """Add request interceptor."""
        self.interceptors.append(interceptor)
    
    def proxy_request(self, service_name: str, request: Dict,
                     client_ip: str = None) -> Dict:
        """Proxy request through mesh."""
        # Apply interceptors
        for interceptor in self.interceptors:
            request = interceptor(request)
        
        # Route to instance
        instance = self.mesh.route_request(service_name, request, client_ip)
        
        if not instance:
            return {'error': 'No healthy instances', 'status': 503}
        
        # Return instance info
        return {
            'target': f"{instance.host}:{instance.port}",
            'instance_id': instance.instance_id,
            'status': 200
        }

# Example usage
if __name__ == "__main__":
    mesh = ServiceMeshV2()
    
    # Create service
    service = Service(
        name="face-api",
        namespace="default",
        version="v1",
        replicas=3
    )
    mesh.register_service(service, LoadBalancingStrategy.ROUND_ROBIN)
    
    # Add instances
    for i in range(3):
        instance = ServiceInstance(
            instance_id=f"instance-{i}",
            host="localhost",
            port=8000 + i
        )
        mesh.add_instance("face-api", instance)
    
    # Route requests
    for _ in range(5):
        target = mesh.route_request("face-api", {})
        print(f"Routed to: {target.host}:{target.port}")
    
    # Get status
    status = mesh.get_mesh_status()
    print(f"\nMesh Status: {json.dumps(status, indent=2)}")
