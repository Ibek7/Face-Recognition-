# Service Factory & Lifecycle Management

import threading
import time
from typing import Dict, Any, Optional, Callable, Type, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json

class ServiceLifecycle(Enum):
    """Service lifecycle states."""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"

@dataclass
class ServiceConfig:
    """Service configuration."""
    name: str
    service_type: Type
    factory: Optional[Callable] = None
    auto_start: bool = True
    startup_timeout: float = 30.0
    shutdown_timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'service_type': self.service_type.__name__ if self.service_type else '',
            'auto_start': self.auto_start,
            'dependencies': self.dependencies,
            'metadata': self.metadata
        }

class IService(ABC):
    """Base service interface."""
    
    @abstractmethod
    def start(self) -> None:
        """Start service."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop service."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check service health."""
        pass

@dataclass
class ServiceMetrics:
    """Service metrics."""
    start_time: Optional[float] = None
    stop_time: Optional[float] = None
    total_requests: int = 0
    total_errors: int = 0
    avg_response_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'start_time': self.start_time,
            'stop_time': self.stop_time,
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'avg_response_time': self.avg_response_time
        }

@dataclass
class ServiceInstance:
    """Service instance wrapper."""
    config: ServiceConfig
    instance: Optional[Any] = None
    lifecycle: ServiceLifecycle = ServiceLifecycle.NOT_STARTED
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)
    error_message: Optional[str] = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.config.name,
            'lifecycle': self.lifecycle.value,
            'error': self.error_message,
            'metrics': self.metrics.to_dict(),
            'config': self.config.to_dict()
        }

class ServiceFactory:
    """Factory for creating services."""
    
    def __init__(self):
        self.creators: Dict[Type, Callable] = {}
        self.lock = threading.RLock()
    
    def register_creator(self, service_type: Type, 
                        creator: Callable) -> None:
        """Register service creator."""
        with self.lock:
            self.creators[service_type] = creator
    
    def create(self, service_type: Type, **kwargs) -> Any:
        """Create service instance."""
        with self.lock:
            if service_type in self.creators:
                creator = self.creators[service_type]
                return creator(**kwargs)
            
            # Default: instantiate directly
            return service_type(**kwargs)

class ServiceRegistry:
    """Registry for managing service instances."""
    
    def __init__(self):
        self.services: Dict[str, ServiceInstance] = {}
        self.factory = ServiceFactory()
        self.lock = threading.RLock()
    
    def register(self, config: ServiceConfig) -> ServiceInstance:
        """Register service."""
        with self.lock:
            if config.name in self.services:
                raise ValueError(f"Service {config.name} already registered")
            
            service_instance = ServiceInstance(config=config)
            self.services[config.name] = service_instance
            
            return service_instance
    
    def unregister(self, name: str) -> bool:
        """Unregister service."""
        with self.lock:
            if name in self.services:
                del self.services[name]
                return True
            return False
    
    def get(self, name: str) -> Optional[ServiceInstance]:
        """Get service instance."""
        with self.lock:
            return self.services.get(name)
    
    def get_all(self) -> List[ServiceInstance]:
        """Get all services."""
        with self.lock:
            return list(self.services.values())

class ServiceLifecycleManager:
    """Manage service lifecycle."""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.lock = threading.RLock()
    
    def start_service(self, name: str) -> bool:
        """Start service."""
        service_instance = self.registry.get(name)
        if not service_instance:
            return False
        
        with service_instance.lock:
            if service_instance.lifecycle == ServiceLifecycle.RUNNING:
                return True
            
            try:
                service_instance.lifecycle = ServiceLifecycle.INITIALIZING
                service_instance.metrics.start_time = time.time()
                
                # Check dependencies
                for dep_name in service_instance.config.dependencies:
                    dep = self.registry.get(dep_name)
                    if not dep or dep.lifecycle != ServiceLifecycle.RUNNING:
                        raise Exception(f"Dependency {dep_name} not running")
                
                # Create instance
                if service_instance.config.factory:
                    service_instance.instance = service_instance.config.factory()
                else:
                    service_instance.instance = self.registry.factory.create(
                        service_instance.config.service_type
                    )
                
                # Start service
                if isinstance(service_instance.instance, IService):
                    service_instance.instance.start()
                
                service_instance.lifecycle = ServiceLifecycle.RUNNING
                return True
            
            except Exception as e:
                service_instance.lifecycle = ServiceLifecycle.FAILED
                service_instance.error_message = str(e)
                return False
    
    def stop_service(self, name: str) -> bool:
        """Stop service."""
        service_instance = self.registry.get(name)
        if not service_instance:
            return False
        
        with service_instance.lock:
            if service_instance.lifecycle == ServiceLifecycle.STOPPED:
                return True
            
            try:
                service_instance.lifecycle = ServiceLifecycle.STOPPING
                
                # Stop service
                if isinstance(service_instance.instance, IService):
                    service_instance.instance.stop()
                
                service_instance.lifecycle = ServiceLifecycle.STOPPED
                service_instance.metrics.stop_time = time.time()
                return True
            
            except Exception as e:
                service_instance.lifecycle = ServiceLifecycle.FAILED
                service_instance.error_message = str(e)
                return False
    
    def start_all(self) -> Dict[str, bool]:
        """Start all services."""
        results = {}
        
        # Start in dependency order
        for service_instance in self.registry.get_all():
            if service_instance.config.auto_start:
                results[service_instance.config.name] = self.start_service(
                    service_instance.config.name
                )
        
        return results
    
    def stop_all(self) -> Dict[str, bool]:
        """Stop all services."""
        results = {}
        
        # Stop in reverse order
        for service_instance in reversed(self.registry.get_all()):
            results[service_instance.config.name] = self.stop_service(
                service_instance.config.name
            )
        
        return results
    
    def restart_service(self, name: str) -> bool:
        """Restart service."""
        self.stop_service(name)
        time.sleep(0.5)
        return self.start_service(name)

class ServiceHealthMonitor:
    """Monitor service health."""
    
    def __init__(self, registry: ServiceRegistry, 
                 check_interval: float = 10.0):
        self.registry = registry
        self.check_interval = check_interval
        self.running = False
    
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        self.running = True
        
        def monitor():
            while self.running:
                for service_instance in self.registry.get_all():
                    if service_instance.lifecycle == ServiceLifecycle.RUNNING:
                        try:
                            if isinstance(service_instance.instance, IService):
                                healthy = service_instance.instance.health_check()
                                if not healthy:
                                    service_instance.error_message = "Health check failed"
                        except Exception as e:
                            service_instance.error_message = str(e)
                
                time.sleep(self.check_interval)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.running = False

class ServiceBootstrapper:
    """Bootstrap application services."""
    
    def __init__(self):
        self.registry = ServiceRegistry()
        self.lifecycle_manager = ServiceLifecycleManager(self.registry)
        self.health_monitor = ServiceHealthMonitor(self.registry)
    
    def configure_service(self, config: ServiceConfig) -> ServiceInstance:
        """Configure service."""
        return self.registry.register(config)
    
    def start(self) -> Dict[str, bool]:
        """Start all services."""
        results = self.lifecycle_manager.start_all()
        self.health_monitor.start_monitoring()
        return results
    
    def stop(self) -> Dict[str, bool]:
        """Stop all services."""
        self.health_monitor.stop_monitoring()
        return self.lifecycle_manager.stop_all()
    
    def get_status(self) -> Dict:
        """Get status of all services."""
        return {
            'services': [
                service.to_dict() 
                for service in self.registry.get_all()
            ]
        }

# Example usage
if __name__ == "__main__":
    # Define example service
    class DatabaseService(IService):
        def start(self) -> None:
            print("Database service starting...")
            time.sleep(0.5)
            print("Database service started")
        
        def stop(self) -> None:
            print("Database service stopping...")
            time.sleep(0.5)
            print("Database service stopped")
        
        def health_check(self) -> bool:
            return True
    
    class CacheService(IService):
        def start(self) -> None:
            print("Cache service starting...")
        
        def stop(self) -> None:
            print("Cache service stopping...")
        
        def health_check(self) -> bool:
            return True
    
    # Bootstrap
    bootstrapper = ServiceBootstrapper()
    
    # Configure services
    db_config = ServiceConfig(
        name="database",
        service_type=DatabaseService,
        auto_start=True
    )
    
    cache_config = ServiceConfig(
        name="cache",
        service_type=CacheService,
        auto_start=True,
        dependencies=["database"]
    )
    
    bootstrapper.configure_service(db_config)
    bootstrapper.configure_service(cache_config)
    
    # Start services
    results = bootstrapper.start()
    print(f"Start Results: {results}")
    
    # Get status
    status = bootstrapper.get_status()
    print(f"\nService Status:")
    print(json.dumps(status, indent=2))
    
    # Stop services
    results = bootstrapper.stop()
    print(f"\nStop Results: {results}")
