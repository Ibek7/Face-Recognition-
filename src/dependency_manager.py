"""
Dependency injection container for service lifecycle management.

Provides dependency injection, service registration, and lifecycle hooks.
"""

import asyncio
import inspect
from typing import Any, Callable, Dict, Optional, Type, TypeVar, get_type_hints
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceScope(str, Enum):
    """Service lifecycle scopes."""
    
    SINGLETON = "singleton"  # Single instance
    TRANSIENT = "transient"  # New instance per request
    SCOPED = "scoped"  # Single instance per scope


class ServiceDescriptor:
    """Describes a registered service."""
    
    def __init__(
        self,
        service_type: Type,
        implementation: Optional[Type] = None,
        factory: Optional[Callable] = None,
        instance: Optional[Any] = None,
        scope: ServiceScope = ServiceScope.SINGLETON
    ):
        """
        Initialize service descriptor.
        
        Args:
            service_type: Service type/interface
            implementation: Implementation class
            factory: Factory function
            instance: Singleton instance
            scope: Service scope
        """
        self.service_type = service_type
        self.implementation = implementation
        self.factory = factory
        self.instance = instance
        self.scope = scope
        self.initialized = False


class DependencyContainer:
    """Dependency injection container."""
    
    def __init__(self):
        """Initialize dependency container."""
        self.services: Dict[Type, ServiceDescriptor] = {}
        self.scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self.current_scope: Optional[str] = None
    
    def register(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None,
        scope: ServiceScope = ServiceScope.SINGLETON
    ):
        """
        Register service implementation.
        
        Args:
            service_type: Service type/interface
            implementation: Implementation class
            scope: Service scope
        """
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation or service_type,
            scope=scope
        )
        
        self.services[service_type] = descriptor
        
        logger.info(
            f"Registered service: {service_type.__name__} "
            f"({scope.value})"
        )
    
    def register_instance(
        self,
        service_type: Type[T],
        instance: T
    ):
        """
        Register singleton instance.
        
        Args:
            service_type: Service type
            instance: Service instance
        """
        descriptor = ServiceDescriptor(
            service_type=service_type,
            instance=instance,
            scope=ServiceScope.SINGLETON
        )
        descriptor.initialized = True
        
        self.services[service_type] = descriptor
        
        logger.info(f"Registered instance: {service_type.__name__}")
    
    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[..., T],
        scope: ServiceScope = ServiceScope.TRANSIENT
    ):
        """
        Register service factory.
        
        Args:
            service_type: Service type
            factory: Factory function
            scope: Service scope
        """
        descriptor = ServiceDescriptor(
            service_type=service_type,
            factory=factory,
            scope=scope
        )
        
        self.services[service_type] = descriptor
        
        logger.info(
            f"Registered factory: {service_type.__name__} "
            f"({scope.value})"
        )
    
    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve service instance.
        
        Args:
            service_type: Service type to resolve
        
        Returns:
            Service instance
        
        Raises:
            ValueError: If service not registered
        """
        if service_type not in self.services:
            raise ValueError(f"Service not registered: {service_type.__name__}")
        
        descriptor = self.services[service_type]
        
        # Return singleton instance
        if descriptor.scope == ServiceScope.SINGLETON:
            if descriptor.instance is not None:
                return descriptor.instance
            
            # Create and cache singleton
            instance = self._create_instance(descriptor)
            descriptor.instance = instance
            descriptor.initialized = True
            
            return instance
        
        # Return scoped instance
        elif descriptor.scope == ServiceScope.SCOPED:
            if not self.current_scope:
                raise ValueError("No active scope")
            
            scope_instances = self.scoped_instances.get(self.current_scope, {})
            
            if service_type in scope_instances:
                return scope_instances[service_type]
            
            # Create scoped instance
            instance = self._create_instance(descriptor)
            scope_instances[service_type] = instance
            self.scoped_instances[self.current_scope] = scope_instances
            
            return instance
        
        # Create transient instance
        else:
            return self._create_instance(descriptor)
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create service instance."""
        # Use factory if provided
        if descriptor.factory:
            if asyncio.iscoroutinefunction(descriptor.factory):
                # Can't handle async in sync context
                raise ValueError("Async factories not supported in sync resolve")
            
            # Resolve factory dependencies
            return self._invoke_with_injection(descriptor.factory)
        
        # Use implementation class
        if descriptor.implementation:
            return self._invoke_with_injection(descriptor.implementation)
        
        raise ValueError(f"Cannot create instance: {descriptor.service_type.__name__}")
    
    def _invoke_with_injection(self, callable_obj: Callable) -> Any:
        """
        Invoke callable with dependency injection.
        
        Args:
            callable_obj: Callable to invoke
        
        Returns:
            Result of invocation
        """
        # Get type hints
        sig = inspect.signature(callable_obj)
        type_hints = get_type_hints(callable_obj)
        
        # Resolve dependencies
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param_name in type_hints:
                param_type = type_hints[param_name]
                
                # Skip non-class types
                if not inspect.isclass(param_type):
                    continue
                
                # Resolve dependency
                if param_type in self.services:
                    kwargs[param_name] = self.resolve(param_type)
        
        # Invoke
        return callable_obj(**kwargs)
    
    async def resolve_async(self, service_type: Type[T]) -> T:
        """
        Resolve service asynchronously.
        
        Args:
            service_type: Service type
        
        Returns:
            Service instance
        """
        # For now, just call sync resolve
        # In future, support async factories
        return self.resolve(service_type)
    
    def create_scope(self, scope_id: Optional[str] = None) -> 'ServiceScope':
        """
        Create dependency scope.
        
        Args:
            scope_id: Scope identifier
        
        Returns:
            Scope context manager
        """
        return DependencyScope(self, scope_id)
    
    async def initialize_all(self):
        """Initialize all singleton services."""
        for service_type, descriptor in self.services.items():
            if descriptor.scope == ServiceScope.SINGLETON and not descriptor.initialized:
                try:
                    self.resolve(service_type)
                    logger.info(f"Initialized service: {service_type.__name__}")
                except Exception as e:
                    logger.error(f"Failed to initialize {service_type.__name__}: {e}")
    
    async def shutdown_all(self):
        """Shutdown all services with cleanup."""
        for service_type, descriptor in self.services.items():
            if descriptor.instance is None:
                continue
            
            # Call shutdown method if exists
            if hasattr(descriptor.instance, 'shutdown'):
                try:
                    shutdown_method = descriptor.instance.shutdown
                    
                    if asyncio.iscoroutinefunction(shutdown_method):
                        await shutdown_method()
                    else:
                        shutdown_method()
                    
                    logger.info(f"Shutdown service: {service_type.__name__}")
                
                except Exception as e:
                    logger.error(f"Error shutting down {service_type.__name__}: {e}")


class DependencyScope:
    """Dependency scope context manager."""
    
    def __init__(self, container: DependencyContainer, scope_id: Optional[str] = None):
        """
        Initialize scope.
        
        Args:
            container: Dependency container
            scope_id: Scope identifier
        """
        self.container = container
        self.scope_id = scope_id or str(id(self))
    
    def __enter__(self):
        """Enter scope."""
        self.container.current_scope = self.scope_id
        self.container.scoped_instances[self.scope_id] = {}
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit scope."""
        # Cleanup scoped instances
        if self.scope_id in self.container.scoped_instances:
            del self.container.scoped_instances[self.scope_id]
        
        self.container.current_scope = None


# Global dependency container
container = DependencyContainer()


# Example usage:
"""
from src.dependency_manager import container, ServiceScope

# Define services
class DatabaseService:
    def __init__(self):
        self.connection = "db_connection"
    
    async def shutdown(self):
        print("Closing database connection")

class UserRepository:
    def __init__(self, db: DatabaseService):
        self.db = db
    
    def get_user(self, user_id: str):
        return {"id": user_id, "db": self.db.connection}

class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo
    
    def get_user(self, user_id: str):
        return self.repo.get_user(user_id)

# Register services
container.register(DatabaseService, scope=ServiceScope.SINGLETON)
container.register(UserRepository, scope=ServiceScope.SINGLETON)
container.register(UserService, scope=ServiceScope.TRANSIENT)

# Initialize all singletons
await container.initialize_all()

# Resolve service (dependencies auto-injected)
user_service = container.resolve(UserService)
user = user_service.get_user("123")
print(user)

# Shutdown
await container.shutdown_all()

# Using scopes
with container.create_scope() as scope:
    scoped_service = container.resolve(ScopedService)
"""
