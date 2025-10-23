# Plugin System & Extensibility Framework

import importlib
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json

class PluginType(Enum):
    """Types of plugins."""
    PROCESSOR = "processor"
    VALIDATOR = "validator"
    EXPORTER = "exporter"
    ANALYZER = "analyzer"
    TRANSFORMER = "transformer"

@dataclass
class PluginMetadata:
    """Plugin metadata."""
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    config_schema: Optional[Dict] = None
    enabled: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'author': self.author,
            'description': self.description,
            'plugin_type': self.plugin_type.value,
            'dependencies': self.dependencies,
            'enabled': self.enabled
        }

class Plugin(ABC):
    """Base class for plugins."""
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.config: Dict[str, Any] = {}
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize plugin. Return True if successful."""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin logic."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Shutdown plugin."""
        pass
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self.metadata
    
    def validate_config(self, config: Dict) -> bool:
        """Validate configuration."""
        return True

class ProcessorPlugin(Plugin):
    """Plugin for processing data."""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data."""
        pass
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute processing."""
        if args:
            return self.process(args[0])
        return None

class ValidatorPlugin(Plugin):
    """Plugin for validation."""
    
    @abstractmethod
    def validate(self, data: Any) -> Tuple[bool, str]:
        """Validate data. Return (is_valid, message)."""
        pass
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute validation."""
        if args:
            return self.validate(args[0])
        return (False, "No data provided")

class PluginRegistry:
    """Registry for plugins."""
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_hooks: Dict[str, List[Plugin]] = {}
        self.lock = threading.RLock()
    
    def register_plugin(self, plugin_id: str, plugin: Plugin) -> bool:
        """Register plugin."""
        with self.lock:
            if plugin_id in self.plugins:
                return False
            
            if not plugin.metadata.enabled:
                return False
            
            self.plugins[plugin_id] = plugin
            
            # Index by type
            plugin_type = plugin.metadata.plugin_type.value
            if plugin_type not in self.plugin_hooks:
                self.plugin_hooks[plugin_type] = []
            
            self.plugin_hooks[plugin_type].append(plugin)
            
            return True
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """Unregister plugin."""
        with self.lock:
            if plugin_id not in self.plugins:
                return False
            
            plugin = self.plugins[plugin_id]
            del self.plugins[plugin_id]
            
            # Remove from hooks
            plugin_type = plugin.metadata.plugin_type.value
            if plugin_type in self.plugin_hooks:
                self.plugin_hooks[plugin_type] = [
                    p for p in self.plugin_hooks[plugin_type] if p != plugin
                ]
            
            return True
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get plugin by ID."""
        with self.lock:
            return self.plugins.get(plugin_id)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[Plugin]:
        """Get plugins by type."""
        with self.lock:
            return self.plugin_hooks.get(plugin_type.value, []).copy()
    
    def list_plugins(self) -> List[Dict]:
        """List all plugins."""
        with self.lock:
            return [p.get_metadata().to_dict() for p in self.plugins.values()]

class PluginLoader:
    """Load plugins from modules."""
    
    @staticmethod
    def load_plugin(module_path: str, class_name: str) -> Optional[Type]:
        """Load plugin class from module."""
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            print(f"Error loading plugin: {e}")
            return None
    
    @staticmethod
    def load_and_register(registry: PluginRegistry, plugin_id: str,
                         module_path: str, class_name: str,
                         config: Dict = None) -> bool:
        """Load and register plugin."""
        plugin_class = PluginLoader.load_plugin(module_path, class_name)
        
        if not plugin_class:
            return False
        
        try:
            metadata = PluginMetadata(
                name=class_name,
                version="1.0.0",
                author="Unknown",
                description="Loaded plugin",
                plugin_type=PluginType.PROCESSOR
            )
            
            plugin = plugin_class(metadata)
            
            if not plugin.initialize(config or {}):
                return False
            
            return registry.register_plugin(plugin_id, plugin)
        except Exception as e:
            print(f"Error instantiating plugin: {e}")
            return False

class PluginHook:
    """Hook point in system for plugins."""
    
    def __init__(self, name: str, registry: PluginRegistry):
        self.name = name
        self.registry = registry
    
    def execute(self, plugin_type: PluginType, *args, **kwargs) -> List[Any]:
        """Execute all plugins of given type."""
        plugins = self.registry.get_plugins_by_type(plugin_type)
        results = []
        
        for plugin in plugins:
            try:
                result = plugin.execute(*args, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error executing plugin {plugin.metadata.name}: {e}")
        
        return results

class PluginChain:
    """Chain plugins for sequential execution."""
    
    def __init__(self):
        self.plugins: List[Plugin] = []
    
    def add_plugin(self, plugin: Plugin) -> 'PluginChain':
        """Add plugin to chain."""
        self.plugins.append(plugin)
        return self
    
    def execute(self, data: Any) -> Any:
        """Execute plugins sequentially."""
        result = data
        
        for plugin in self.plugins:
            try:
                result = plugin.execute(result)
            except Exception as e:
                print(f"Error in plugin chain: {e}")
                return None
        
        return result

class PluginManager:
    """Manage plugins lifecycle."""
    
    def __init__(self):
        self.registry = PluginRegistry()
        self.hooks: Dict[str, PluginHook] = {}
        self.chains: Dict[str, PluginChain] = {}
        self.lock = threading.RLock()
        self.event_log: List[Dict] = []
    
    def register_hook(self, hook_name: str) -> PluginHook:
        """Register hook."""
        with self.lock:
            if hook_name not in self.hooks:
                self.hooks[hook_name] = PluginHook(hook_name, self.registry)
            
            return self.hooks[hook_name]
    
    def create_chain(self, chain_name: str) -> PluginChain:
        """Create plugin chain."""
        with self.lock:
            if chain_name not in self.chains:
                self.chains[chain_name] = PluginChain()
            
            return self.chains[chain_name]
    
    def register_plugin(self, plugin_id: str, plugin: Plugin) -> bool:
        """Register plugin."""
        success = self.registry.register_plugin(plugin_id, plugin)
        
        if success:
            with self.lock:
                self.event_log.append({
                    'timestamp': time.time(),
                    'event': 'plugin_registered',
                    'plugin_id': plugin_id,
                    'plugin_name': plugin.metadata.name
                })
        
        return success
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """Unregister plugin."""
        success = self.registry.unregister_plugin(plugin_id)
        
        if success:
            with self.lock:
                self.event_log.append({
                    'timestamp': time.time(),
                    'event': 'plugin_unregistered',
                    'plugin_id': plugin_id
                })
        
        return success
    
    def get_status(self) -> Dict:
        """Get plugin system status."""
        with self.lock:
            return {
                'plugins_count': len(self.registry.plugins),
                'hooks_count': len(self.hooks),
                'chains_count': len(self.chains),
                'plugins': self.registry.list_plugins(),
                'recent_events': self.event_log[-10:]
            }

# Example plugin
class SimpleProcessorPlugin(ProcessorPlugin):
    """Example processor plugin."""
    
    def initialize(self, config: Dict) -> bool:
        """Initialize."""
        self.config = config
        self.is_initialized = True
        return True
    
    def process(self, data: Any) -> Any:
        """Process data."""
        return f"Processed: {data}"
    
    def shutdown(self):
        """Shutdown."""
        pass

# Example usage
if __name__ == "__main__":
    from typing import Tuple
    
    # Create plugin manager
    manager = PluginManager()
    
    # Create and register plugin
    metadata = PluginMetadata(
        name="SimpleProcessor",
        version="1.0.0",
        author="Demo",
        description="Simple processor",
        plugin_type=PluginType.PROCESSOR
    )
    
    plugin = SimpleProcessorPlugin(metadata)
    plugin.initialize({})
    
    manager.register_plugin("simple_1", plugin)
    
    # Create hook
    hook = manager.register_hook("process_hook")
    
    # Execute
    results = hook.execute(PluginType.PROCESSOR, "test_data")
    print(f"Results: {results}")
    
    # Get status
    status = manager.get_status()
    print(f"\nPlugin System Status:")
    print(json.dumps(status, indent=2))
