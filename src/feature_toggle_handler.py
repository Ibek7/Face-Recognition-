"""
Feature toggle system for gradual rollouts and A/B testing.

Provides feature flags with user targeting and percentage rollouts.
"""

from typing import Dict, List, Optional, Any, Set
from enum import Enum
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


class FeatureStatus(str, Enum):
    """Feature flag status."""
    
    ENABLED = "enabled"
    DISABLED = "disabled"
    CONDITIONAL = "conditional"


class RolloutStrategy(str, Enum):
    """Rollout strategy types."""
    
    ALL = "all"  # All users
    PERCENTAGE = "percentage"  # Percentage of users
    WHITELIST = "whitelist"  # Specific users
    ATTRIBUTES = "attributes"  # Based on user attributes


class FeatureFlag:
    """Feature flag definition."""
    
    def __init__(
        self,
        name: str,
        status: FeatureStatus = FeatureStatus.DISABLED,
        description: Optional[str] = None,
        rollout_strategy: RolloutStrategy = RolloutStrategy.ALL,
        rollout_percentage: int = 100,
        whitelist: Optional[Set[str]] = None,
        attribute_rules: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize feature flag.
        
        Args:
            name: Feature name
            status: Feature status
            description: Feature description
            rollout_strategy: Rollout strategy
            rollout_percentage: Rollout percentage (0-100)
            whitelist: User ID whitelist
            attribute_rules: Attribute-based rules
        """
        self.name = name
        self.status = status
        self.description = description
        self.rollout_strategy = rollout_strategy
        self.rollout_percentage = rollout_percentage
        self.whitelist = whitelist or set()
        self.attribute_rules = attribute_rules or {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def is_enabled_for_user(
        self,
        user_id: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if feature is enabled for user.
        
        Args:
            user_id: User identifier
            attributes: User attributes
        
        Returns:
            True if enabled
        """
        # Global disabled
        if self.status == FeatureStatus.DISABLED:
            return False
        
        # Global enabled
        if self.status == FeatureStatus.ENABLED:
            return True
        
        # Conditional - check strategy
        if self.rollout_strategy == RolloutStrategy.ALL:
            return True
        
        elif self.rollout_strategy == RolloutStrategy.WHITELIST:
            return user_id in self.whitelist
        
        elif self.rollout_strategy == RolloutStrategy.PERCENTAGE:
            return self._check_percentage_rollout(user_id)
        
        elif self.rollout_strategy == RolloutStrategy.ATTRIBUTES:
            return self._check_attribute_rules(attributes or {})
        
        return False
    
    def _check_percentage_rollout(self, user_id: str) -> bool:
        """Check if user is in percentage rollout."""
        # Hash user ID to get consistent assignment
        hash_value = int(
            hashlib.md5(f"{self.name}:{user_id}".encode()).hexdigest(),
            16
        )
        
        # Map to 0-100 range
        bucket = hash_value % 100
        
        return bucket < self.rollout_percentage
    
    def _check_attribute_rules(self, attributes: Dict[str, Any]) -> bool:
        """Check if user attributes match rules."""
        for key, expected_value in self.attribute_rules.items():
            if key not in attributes:
                return False
            
            actual_value = attributes[key]
            
            # Handle list of allowed values
            if isinstance(expected_value, list):
                if actual_value not in expected_value:
                    return False
            else:
                if actual_value != expected_value:
                    return False
        
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "description": self.description,
            "rollout_strategy": self.rollout_strategy.value,
            "rollout_percentage": self.rollout_percentage,
            "whitelist": list(self.whitelist),
            "attribute_rules": self.attribute_rules,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class FeatureToggleManager:
    """Manage feature toggles."""
    
    def __init__(self):
        """Initialize feature toggle manager."""
        self.flags: Dict[str, FeatureFlag] = {}
        self.evaluation_count = 0
    
    def create_flag(
        self,
        name: str,
        status: FeatureStatus = FeatureStatus.DISABLED,
        **kwargs
    ):
        """
        Create feature flag.
        
        Args:
            name: Feature name
            status: Initial status
            **kwargs: Additional flag parameters
        """
        flag = FeatureFlag(name=name, status=status, **kwargs)
        self.flags[name] = flag
        
        logger.info(f"Created feature flag: {name} ({status.value})")
    
    def update_flag(
        self,
        name: str,
        status: Optional[FeatureStatus] = None,
        rollout_percentage: Optional[int] = None,
        **kwargs
    ):
        """
        Update feature flag.
        
        Args:
            name: Feature name
            status: New status
            rollout_percentage: New rollout percentage
            **kwargs: Additional parameters to update
        """
        if name not in self.flags:
            raise ValueError(f"Feature flag not found: {name}")
        
        flag = self.flags[name]
        
        if status:
            flag.status = status
        
        if rollout_percentage is not None:
            flag.rollout_percentage = rollout_percentage
        
        for key, value in kwargs.items():
            if hasattr(flag, key):
                setattr(flag, key, value)
        
        flag.updated_at = datetime.utcnow()
        
        logger.info(f"Updated feature flag: {name}")
    
    def delete_flag(self, name: str):
        """Delete feature flag."""
        if name in self.flags:
            del self.flags[name]
            logger.info(f"Deleted feature flag: {name}")
    
    def is_enabled(
        self,
        name: str,
        user_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        default: bool = False
    ) -> bool:
        """
        Check if feature is enabled.
        
        Args:
            name: Feature name
            user_id: User identifier
            attributes: User attributes
            default: Default value if flag not found
        
        Returns:
            True if enabled
        """
        self.evaluation_count += 1
        
        if name not in self.flags:
            logger.warning(f"Feature flag not found: {name}, using default: {default}")
            return default
        
        flag = self.flags[name]
        
        if user_id:
            return flag.is_enabled_for_user(user_id, attributes)
        else:
            # No user context - check global status
            return flag.status == FeatureStatus.ENABLED
    
    def get_flag(self, name: str) -> Optional[FeatureFlag]:
        """Get feature flag."""
        return self.flags.get(name)
    
    def list_flags(self) -> List[dict]:
        """List all feature flags."""
        return [flag.to_dict() for flag in self.flags.values()]
    
    def get_enabled_features(
        self,
        user_id: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Get list of enabled features for user.
        
        Args:
            user_id: User identifier
            attributes: User attributes
        
        Returns:
            List of enabled feature names
        """
        enabled = []
        
        for name, flag in self.flags.items():
            if flag.is_enabled_for_user(user_id, attributes):
                enabled.append(name)
        
        return enabled
    
    def get_stats(self) -> dict:
        """Get feature flag statistics."""
        return {
            "total_flags": len(self.flags),
            "enabled_flags": sum(
                1 for f in self.flags.values()
                if f.status == FeatureStatus.ENABLED
            ),
            "disabled_flags": sum(
                1 for f in self.flags.values()
                if f.status == FeatureStatus.DISABLED
            ),
            "conditional_flags": sum(
                1 for f in self.flags.values()
                if f.status == FeatureStatus.CONDITIONAL
            ),
            "evaluation_count": self.evaluation_count
        }


# Global feature toggle manager
feature_toggles = FeatureToggleManager()


# Example usage:
"""
from fastapi import FastAPI
from src.feature_toggle_handler import feature_toggles, FeatureStatus, RolloutStrategy

app = FastAPI()

@app.on_event("startup")
async def startup():
    # Create feature flags
    feature_toggles.create_flag(
        name="new_ui",
        status=FeatureStatus.CONDITIONAL,
        description="New UI redesign",
        rollout_strategy=RolloutStrategy.PERCENTAGE,
        rollout_percentage=10  # 10% rollout
    )
    
    feature_toggles.create_flag(
        name="advanced_analytics",
        status=FeatureStatus.CONDITIONAL,
        rollout_strategy=RolloutStrategy.WHITELIST,
        whitelist={"user123", "user456"}
    )

@app.get("/api/data")
async def get_data(user_id: str):
    if feature_toggles.is_enabled("new_ui", user_id):
        return {"version": "v2"}
    return {"version": "v1"}

@app.get("/features")
async def list_features(user_id: str):
    enabled = feature_toggles.get_enabled_features(user_id)
    return {"enabled_features": enabled}
"""
