# Dependency Injection Container & IoC Framework

import threading
import inspect
from typing import Dict, Any, Optional, Callable, Type, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

class LifecycleScope(Enum):
    """Dependency lifecycle scopes."""
    SINGLETON = "singleton"  # Single instance per container
    TRANSIENT = "transient"  # New instance every time
    SCOPED = "scoped"  # Single instance per scope

@dataclass
class DependencyDescriptor:
    """Description of dependency."""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    scope: LifecycleScope = LifecycleScope.TRANSIENT
    dependencies: List[str] = field(default_factory=list)
    initialized: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'service': self.service_type.__name__ if self.service_type else '',
            'scope': self.scope.value,
            'initialized': self.initialized
        }

class ServiceProvider:
    """Provide service instances."""
    
    def __init__(self, container: 'DIContainer'):
        self.container = container
        self.lock = threading.RLock()
    
    def get_service(self, service_type: Type) -> Optional[Any]:
        """Get service instance."""
        return self.container.resolve(service_type)
    
    def get_services(self, service_type: Type) -> List[Any]:
        """Get all services of type."""
        return self.container.resolve_all(service_type)

class DIContainer:
    """Dependency Injection Container."""
    
    def __init__(self):
        self.services: Dict[str, DependencyDescriptor] = {}
        self.factories: Dict[str, Callable] = {}
        self.singletons: Dict[str, Any] = {}
        self.scopes: Dict[str, Dict] = {}
        self.lock = threading.RLock()
    
    def register(self, service_type: Type, implementation_type: Type = None,
                scope: LifecycleScope = LifecycleScope.TRANSIENT,
                factory: Callable = None, instance: Any = None) -> 'DIContainer':
        """Register service."""
        service_key = service_type.__name__
        
        with self.lock:
            descriptor = DependencyDescriptor(
                service_type=service_type,
                implementation_type=implementation_type or service_type,
                factory=factory,
                instance=instance,
                scope=scope
            )
            
            self.services[service_key] = descriptor
            
            # Register factory if provided
            if factory:
                self.factories[service_key] = factory
            
            # Register singleton instance
            if scope == LifecycleScope.SINGLETON and instance:
                self.singletons[service_key] = instance
        
        return self
    
    def register_singleton(self, service_type: Type, instance: Any = None) -> 'DIContainer':
        """Register singleton service."""
        return self.register(service_type, instance=instance, 
                           scope=LifecycleScope.SINGLETON)
    
    def register_transient(self, service_type: Type, 
                          implementation_type: Type = None) -> 'DIContainer':
        """Register transient service."""
        return self.register(service_type, implementation_type=implementation_type,
                           scope=LifecycleScope.TRANSIENT)
    
    def register_factory(self, service_type: Type, factory: Callable,
                        scope: LifecycleScope = LifecycleScope.TRANSIENT) -> 'DIContainer':
        """Register service via factory."""
        return self.register(service_type, factory=factory, scope=scope)
    
    def resolve(self, service_type: Type) -> Optional[Any]:
        """Resolve service instance."""
        service_key = service_type.__name__
        
        with self.lock:
            if service_key not in self.services:
                return None
            
            descriptor = self.services[service_key]
            
            # Return singleton
            if descriptor.scope == LifecycleScope.SINGLETON:
                if service_key in self.singletons:
                    return self.singletons[service_key]
                
                # Create singleton
                instance = self._create_instance(descriptor)
                self.singletons[service_key] = instance
                return instance
            
            # Create new instance
            return self._create_instance(descriptor)
    
    def resolve_all(self, service_type: Type) -> List[Any]:
        """Resolve all services of type."""
        result = []
        
        with self.lock:
            for descriptor in self.services.values():
                if descriptor.service_type == service_type or \
                   descriptor.implementation_type == service_type:
                    instance = self.resolve(descriptor.service_type)
                    if instance:
                        result.append(instance)
        
        return result
    
    def _create_instance(self, descriptor: DependencyDescriptor) -> Optional[Any]:
        """Create service instance."""
        # Use factory if provided
        if descriptor.factory:
            return descriptor.factory()
        
        # Create via constructor
        implementation_type = descriptor.implementation_type
        
        try:
            # Get constructor parameters
            sig = inspect.signature(implementation_type.__init__)
            params = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                # Try to resolve parameter
                if param.annotation != inspect.Parameter.empty:
                    param_type = param.annotation
                    resolved = self.resolve(param_type)
                    
                    if resolved:
                        params[param_name] = resolved
            
            # Create instance
            return implementation_type(**params)
        
        except Exception as e:
            print(f"Error creating instance of {implementation_type}: {e}")
            return None
    
    def get_service_provider(self) -> ServiceProvider:
        """Get service provider."""
        return ServiceProvider(self)
    
    def get_status(self) -> Dict:
        """Get container status."""
        with self.lock:
            return {
                'registered_services': len(self.services),
                'singletons': len(self.singletons),
                'services': {
                    key: descriptor.to_dict()
                    for key, descriptor in self.services.items()
                }
            }

class ServiceLocator:
    """Service locator pattern (use sparingly)."""
    
    _instance: Optional['ServiceLocator'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.container = DIContainer()
        
        return cls._instance
    
    def get_service(self, service_type: Type) -> Optional[Any]:
        """Get service from locator."""
        return self.container.resolve(service_type)
    
    def register_service(self, service_type: Type, 
                        implementation_type: Type = None) -> None:
        """Register service in locator."""
        self.container.register(service_type, implementation_type)

class AutowiredClass:
    """Base class for auto-wired dependencies."""
    
    _container: Optional[DIContainer] = None
    
    @classmethod
    def set_container(cls, container: DIContainer):
        """Set DI container."""
        cls._container = container
    
    def get_dependency(self, dependency_type: Type) -> Optional[Any]:
        """Get dependency."""
        if self._container:
            return self._container.resolve(dependency_type)
        return None

class DIModuleLoader:
    """Load DI modules/configurations."""
    
    def __init__(self, container: DIContainer):
        self.container = container
        self.modules: Dict[str, Callable] = {}
    
    def register_module(self, module_name: str, module_config: Callable) -> None:
        """Register DI module."""
        self.modules[module_name] = module_config
    
    def load_modules(self) -> None:
        """Load all registered modules."""
        for module_name, module_config in self.modules.items():
            try:
                module_config(self.container)
            except Exception as e:
                print(f"Error loading module {module_name}: {e}")

# Example usage
if __name__ == "__main__":
    # Define services
    class IUserRepository:
        def get_user(self, user_id: int):
            pass
    
    class UserRepository(IUserRepository):
        def get_user(self, user_id: int):
            return {"id": user_id, "name": "Test User"}
    
    class IUserService:
        def get_user(self, user_id: int):
            pass
    
    class UserService(IUserService):
        def __init__(self, repository: IUserRepository):
            self.repository = repository
        
        def get_user(self, user_id: int):
            return self.repository.get_user(user_id)
    
    # Setup DI container
    container = DIContainer()
    container.register_singleton(IUserRepository, UserRepository())
    container.register_transient(IUserService, UserService)
    
    # Get service
    user_service = container.resolve(IUserService)
    user = user_service.get_user(1)
    
    print(f"User: {user}")
    
    # Get container status
    status = container.get_status()
    print(f"\nContainer Status:")
    import json
    print(json.dumps(status, indent=2))
