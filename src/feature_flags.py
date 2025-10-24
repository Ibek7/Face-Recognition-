# Feature Flags & Dynamic Configuration System

import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

class RolloutStrategy(Enum):
    """Feature rollout strategies."""
    ALL_USERS = "all_users"
    PERCENTAGE = "percentage"
    CANARY = "canary"
    BETA_USERS = "beta_users"
    GRADUAL = "gradual"

@dataclass
class FeatureFlag:
    """Feature flag definition."""
    name: str
    enabled: bool
    rollout_strategy: RolloutStrategy = RolloutStrategy.ALL_USERS
    rollout_percentage: float = 100.0
    beta_users: List[str] = None
    description: str = ""
    created_at: float = None
    updated_at: float = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'rollout_strategy': self.rollout_strategy.value,
            'rollout_percentage': self.rollout_percentage,
            'beta_users': self.beta_users or [],
            'description': self.description,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

class FeatureFlagManager:
    """Manage feature flags."""
    
    def __init__(self):
        self.flags: Dict[str, FeatureFlag] = {}
        self.lock = threading.RLock()
        self.change_listeners: List[Callable] = []
    
    def create_flag(self, name: str, enabled: bool = False,
                   description: str = "",
                   strategy: RolloutStrategy = RolloutStrategy.ALL_USERS,
                   percentage: float = 100.0) -> FeatureFlag:
        """Create feature flag."""
        import time
        
        flag = FeatureFlag(
            name=name,
            enabled=enabled,
            rollout_strategy=strategy,
            rollout_percentage=percentage,
            description=description,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        with self.lock:
            self.flags[name] = flag
            self._notify_listeners('create', flag)
        
        return flag
    
    def update_flag(self, name: str, enabled: bool = None,
                   percentage: float = None,
                   strategy: RolloutStrategy = None) -> Optional[FeatureFlag]:
        """Update feature flag."""
        import time
        
        with self.lock:
            if name not in self.flags:
                return None
            
            flag = self.flags[name]
            
            if enabled is not None:
                flag.enabled = enabled
            if percentage is not None:
                flag.rollout_percentage = percentage
            if strategy is not None:
                flag.rollout_strategy = strategy
            
            flag.updated_at = time.time()
            self._notify_listeners('update', flag)
        
        return flag
    
    def is_enabled(self, flag_name: str, user_id: str = None) -> bool:
        """Check if flag is enabled for user."""
        with self.lock:
            if flag_name not in self.flags:
                return False
            
            flag = self.flags[flag_name]
            
            if not flag.enabled:
                return False
            
            # Check rollout strategy
            if flag.rollout_strategy == RolloutStrategy.ALL_USERS:
                return True
            
            elif flag.rollout_strategy == RolloutStrategy.PERCENTAGE:
                if user_id is None:
                    return False
                
                user_hash = hash(user_id) % 100
                return user_hash < flag.rollout_percentage
            
            elif flag.rollout_strategy == RolloutStrategy.BETA_USERS:
                return user_id in (flag.beta_users or [])
            
            elif flag.rollout_strategy == RolloutStrategy.CANARY:
                if user_id is None:
                    return False
                # Canary: only specific users
                canary_users = flag.beta_users or []
                return user_id in canary_users
            
            elif flag.rollout_strategy == RolloutStrategy.GRADUAL:
                if user_id is None:
                    return False
                user_hash = hash(user_id) % 1000
                return user_hash < int(flag.rollout_percentage * 10)
            
            return False
    
    def get_flag(self, name: str) -> Optional[FeatureFlag]:
        """Get flag by name."""
        with self.lock:
            return self.flags.get(name)
    
    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """Get all flags."""
        with self.lock:
            return {name: flag for name, flag in self.flags.items()}
    
    def delete_flag(self, name: str) -> bool:
        """Delete flag."""
        with self.lock:
            if name in self.flags:
                flag = self.flags.pop(name)
                self._notify_listeners('delete', flag)
                return True
            return False
    
    def add_listener(self, listener: Callable) -> None:
        """Add change listener."""
        with self.lock:
            self.change_listeners.append(listener)
    
    def _notify_listeners(self, action: str, flag: FeatureFlag) -> None:
        """Notify listeners of changes."""
        for listener in self.change_listeners:
            try:
                listener(action, flag)
            except Exception as e:
                print(f"Error in listener: {e}")

class DynamicConfiguration:
    """Manage dynamic configuration."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.lock = threading.RLock()
        self.watchers: Dict[str, List[Callable]] = {}
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        with self.lock:
            old_value = self.config.get(key)
            self.config[key] = value
            
            # Notify watchers
            if key in self.watchers:
                for watcher in self.watchers[key]:
                    try:
                        watcher(key, old_value, value)
                    except Exception as e:
                        print(f"Error in watcher: {e}")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        with self.lock:
            return self.config.get(key, default)
    
    def watch_config(self, key: str, callback: Callable) -> None:
        """Watch configuration key."""
        with self.lock:
            if key not in self.watchers:
                self.watchers[key] = []
            self.watchers[key].append(callback)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration."""
        with self.lock:
            return self.config.copy()

class ExperimentManager:
    """Manage A/B experiments."""
    
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}
        self.user_assignments: Dict[str, Dict] = {}
        self.lock = threading.RLock()
    
    def create_experiment(self, name: str, variants: List[str],
                         traffic_split: Dict[str, float] = None) -> Dict:
        """Create experiment."""
        if traffic_split is None:
            traffic_split = {v: 1.0 / len(variants) for v in variants}
        
        experiment = {
            'name': name,
            'variants': variants,
            'traffic_split': traffic_split,
            'created_at': datetime.now().isoformat()
        }
        
        with self.lock:
            self.experiments[name] = experiment
        
        return experiment
    
    def assign_variant(self, experiment_name: str, user_id: str) -> Optional[str]:
        """Assign user to variant."""
        with self.lock:
            if experiment_name not in self.experiments:
                return None
            
            # Check if already assigned
            if user_id in self.user_assignments:
                if experiment_name in self.user_assignments[user_id]:
                    return self.user_assignments[user_id][experiment_name]
            
            experiment = self.experiments[experiment_name]
            
            # Assign based on hash and traffic split
            user_hash = hash(user_id) % 100
            cumulative = 0
            
            for variant, percentage in experiment['traffic_split'].items():
                cumulative += percentage * 100
                if user_hash < cumulative:
                    # Store assignment
                    if user_id not in self.user_assignments:
                        self.user_assignments[user_id] = {}
                    self.user_assignments[user_id][experiment_name] = variant
                    return variant
            
            return experiment['variants'][0]
    
    def get_variant(self, experiment_name: str, user_id: str) -> Optional[str]:
        """Get user's variant."""
        with self.lock:
            if user_id in self.user_assignments:
                return self.user_assignments[user_id].get(experiment_name)
            return None

class FeatureContext:
    """Feature evaluation context."""
    
    def __init__(self, flag_manager: FeatureFlagManager,
                 config: DynamicConfiguration = None):
        self.flag_manager = flag_manager
        self.config = config or DynamicConfiguration()
    
    def is_feature_enabled(self, feature_name: str, user_id: str = None) -> bool:
        """Check if feature is enabled."""
        return self.flag_manager.is_enabled(feature_name, user_id)
    
    def get_feature_config(self, feature_name: str) -> Dict:
        """Get feature configuration."""
        return self.flag_manager.get_flag(feature_name).to_dict() \
               if self.flag_manager.get_flag(feature_name) else {}

# Example usage
if __name__ == "__main__":
    manager = FeatureFlagManager()
    config = DynamicConfiguration()
    
    # Create flags
    manager.create_flag("new_ui", enabled=True, 
                       strategy=RolloutStrategy.PERCENTAGE,
                       percentage=50.0)
    
    manager.create_flag("experimental_algo", enabled=False)
    
    # Check if enabled
    print(f"new_ui for user1: {manager.is_enabled('new_ui', 'user1')}")
    print(f"new_ui for user2: {manager.is_enabled('new_ui', 'user2')}")
    
    # Set configuration
    config.set_config("max_batch_size", 1000)
    print(f"max_batch_size: {config.get_config('max_batch_size')}")
    
    # Create experiment
    experiment = ExperimentManager()
    experiment.create_experiment("algo_v1_vs_v2", ["algo_v1", "algo_v2"])
    
    print(f"user1 variant: {experiment.assign_variant('algo_v1_vs_v2', 'user1')}")
    print(f"user2 variant: {experiment.assign_variant('algo_v1_vs_v2', 'user2')}")
