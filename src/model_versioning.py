"""
Model Versioning System

ML model version control with:
- Model registry
- Version tracking
- Model lineage
- A/B testing support
- Model promotion (dev -> staging -> production)
- Model comparison
- Rollback support

Features:
- Metadata tracking (accuracy, training data, hyperparameters)
- Model artifact storage
- Version tags and aliases
- Model performance metrics
"""

import logging
import json
import shutil
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Model deployment stage"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelMetadata:
    """Model metadata"""
    name: str
    version: str
    stage: ModelStage
    framework: str  # "pytorch", "tensorflow", "onnx"
    created_at: datetime
    created_by: str
    
    # Training info
    training_dataset: Optional[str] = None
    training_metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Model info
    model_path: Optional[str] = None
    model_size_bytes: Optional[int] = None
    model_hash: Optional[str] = None
    
    # Performance
    accuracy: Optional[float] = None
    inference_time_ms: Optional[float] = None
    
    # Tags and description
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None
    
    # Lineage
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "stage": self.stage.value,
            "framework": self.framework,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "training_dataset": self.training_dataset,
            "training_metrics": self.training_metrics,
            "hyperparameters": self.hyperparameters,
            "model_path": self.model_path,
            "model_size_bytes": self.model_size_bytes,
            "model_hash": self.model_hash,
            "accuracy": self.accuracy,
            "inference_time_ms": self.inference_time_ms,
            "tags": self.tags,
            "description": self.description,
            "parent_version": self.parent_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        data = data.copy()
        data["stage"] = ModelStage(data["stage"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class ModelRegistry:
    """Model registry for version control"""
    
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.registry_dir / "registry.json"
        
        # Model storage
        self.models_dir = self.registry_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Load registry
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, List[ModelMetadata]]:
        """Load registry from disk"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            registry = {}
            for model_name, versions in data.items():
                registry[model_name] = [
                    ModelMetadata.from_dict(v) for v in versions
                ]
            
            return registry
        
        return {}
    
    def _save_registry(self):
        """Save registry to disk"""
        data = {}
        for model_name, versions in self.registry.items():
            data[model_name] = [v.to_dict() for v in versions]
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate file hash"""
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def register_model(
        self,
        name: str,
        version: str,
        model_path: str,
        framework: str,
        created_by: str,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        training_dataset: Optional[str] = None,
        training_metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        accuracy: Optional[float] = None,
        inference_time_ms: Optional[float] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        parent_version: Optional[str] = None
    ) -> ModelMetadata:
        """
        Register a new model version
        
        Args:
            name: Model name
            version: Version string (e.g., "1.0.0")
            model_path: Path to model file
            framework: Framework name
            created_by: Creator name
            stage: Deployment stage
            training_dataset: Training dataset name
            training_metrics: Training metrics
            hyperparameters: Model hyperparameters
            accuracy: Model accuracy
            inference_time_ms: Average inference time
            tags: Model tags
            description: Model description
            parent_version: Parent version for lineage
        
        Returns:
            ModelMetadata
        """
        # Check if version already exists
        if name in self.registry:
            for existing in self.registry[name]:
                if existing.version == version:
                    raise ValueError(f"Version {version} already exists for model {name}")
        
        # Copy model file
        source_path = Path(model_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_storage_dir = self.models_dir / name / version
        model_storage_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = model_storage_dir / source_path.name
        shutil.copy2(source_path, dest_path)
        
        # Calculate hash
        model_hash = self._calculate_hash(dest_path)
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            stage=stage,
            framework=framework,
            created_at=datetime.now(),
            created_by=created_by,
            training_dataset=training_dataset,
            training_metrics=training_metrics or {},
            hyperparameters=hyperparameters or {},
            model_path=str(dest_path),
            model_size_bytes=dest_path.stat().st_size,
            model_hash=model_hash,
            accuracy=accuracy,
            inference_time_ms=inference_time_ms,
            tags=tags or [],
            description=description,
            parent_version=parent_version
        )
        
        # Add to registry
        if name not in self.registry:
            self.registry[name] = []
        
        self.registry[name].append(metadata)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered model {name} version {version}")
        logger.info(f"  Stage: {stage.value}")
        logger.info(f"  Size: {metadata.model_size_bytes / (1024*1024):.2f} MB")
        logger.info(f"  Hash: {model_hash[:16]}...")
        
        return metadata
    
    def get_model(self, name: str, version: Optional[str] = None, stage: Optional[ModelStage] = None) -> Optional[ModelMetadata]:
        """
        Get model metadata
        
        Args:
            name: Model name
            version: Specific version (if None, returns latest)
            stage: Filter by stage
        
        Returns:
            ModelMetadata or None
        """
        if name not in self.registry:
            return None
        
        versions = self.registry[name]
        
        # Filter by stage
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        if not versions:
            return None
        
        # Get specific version
        if version:
            for v in versions:
                if v.version == version:
                    return v
            return None
        
        # Return latest (most recent)
        return max(versions, key=lambda v: v.created_at)
    
    def list_models(self) -> List[str]:
        """List all model names"""
        return list(self.registry.keys())
    
    def list_versions(self, name: str, stage: Optional[ModelStage] = None) -> List[ModelMetadata]:
        """List all versions of a model"""
        if name not in self.registry:
            return []
        
        versions = self.registry[name]
        
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def promote_model(self, name: str, version: str, target_stage: ModelStage) -> bool:
        """
        Promote model to a different stage
        
        Args:
            name: Model name
            version: Version to promote
            target_stage: Target deployment stage
        
        Returns:
            True if successful
        """
        model = self.get_model(name, version)
        
        if not model:
            logger.error(f"Model {name} version {version} not found")
            return False
        
        # Update stage
        model.stage = target_stage
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Promoted {name} version {version} to {target_stage.value}")
        
        return True
    
    def compare_versions(
        self,
        name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two model versions"""
        model1 = self.get_model(name, version1)
        model2 = self.get_model(name, version2)
        
        if not model1 or not model2:
            raise ValueError("One or both versions not found")
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "differences": {}
        }
        
        # Compare metrics
        if model1.accuracy and model2.accuracy:
            comparison["differences"]["accuracy"] = {
                "version1": model1.accuracy,
                "version2": model2.accuracy,
                "delta": model2.accuracy - model1.accuracy
            }
        
        if model1.inference_time_ms and model2.inference_time_ms:
            comparison["differences"]["inference_time"] = {
                "version1": model1.inference_time_ms,
                "version2": model2.inference_time_ms,
                "delta": model2.inference_time_ms - model1.inference_time_ms
            }
        
        # Compare size
        if model1.model_size_bytes and model2.model_size_bytes:
            comparison["differences"]["size"] = {
                "version1": model1.model_size_bytes,
                "version2": model2.model_size_bytes,
                "delta": model2.model_size_bytes - model1.model_size_bytes
            }
        
        return comparison
    
    def delete_version(self, name: str, version: str) -> bool:
        """Delete a model version"""
        if name not in self.registry:
            return False
        
        # Find and remove version
        self.registry[name] = [
            v for v in self.registry[name] if v.version != version
        ]
        
        # Delete model files
        model_dir = self.models_dir / name / version
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Deleted {name} version {version}")
        
        return True
    
    def get_lineage(self, name: str, version: str) -> List[ModelMetadata]:
        """Get model lineage (ancestry chain)"""
        lineage = []
        
        current = self.get_model(name, version)
        
        while current:
            lineage.append(current)
            
            if current.parent_version:
                current = self.get_model(name, current.parent_version)
            else:
                break
        
        return lineage


# Example usage
if __name__ == "__main__":
    # Initialize registry
    registry = ModelRegistry()
    
    print("Model Registry Demo")
    print("=" * 60)
    print()
    
    # Register a model (example - would use actual model file)
    try:
        # Create dummy model file for demo
        dummy_model = Path("dummy_model.pth")
        dummy_model.write_text("dummy model content")
        
        metadata = registry.register_model(
            name="face_detector",
            version="1.0.0",
            model_path=str(dummy_model),
            framework="pytorch",
            created_by="data_scientist@example.com",
            stage=ModelStage.DEVELOPMENT,
            training_dataset="faces_v1",
            training_metrics={"loss": 0.05, "val_loss": 0.08},
            hyperparameters={"lr": 0.001, "batch_size": 32},
            accuracy=0.95,
            inference_time_ms=15.5,
            tags=["resnet50", "face-detection"],
            description="Initial face detector model"
        )
        
        print("✓ Registered model version 1.0.0")
        print()
        
        # Register v1.1.0 with v1.0.0 as parent
        metadata_v2 = registry.register_model(
            name="face_detector",
            version="1.1.0",
            model_path=str(dummy_model),
            framework="pytorch",
            created_by="data_scientist@example.com",
            stage=ModelStage.DEVELOPMENT,
            training_dataset="faces_v2",
            accuracy=0.97,
            inference_time_ms=14.2,
            parent_version="1.0.0",
            description="Improved model with more training data"
        )
        
        print("✓ Registered model version 1.1.0")
        print()
        
        # List versions
        print("Model versions:")
        versions = registry.list_versions("face_detector")
        for v in versions:
            print(f"  {v.version}: {v.stage.value} (accuracy: {v.accuracy})")
        print()
        
        # Promote to staging
        registry.promote_model("face_detector", "1.1.0", ModelStage.STAGING)
        print("✓ Promoted v1.1.0 to staging")
        print()
        
        # Compare versions
        comparison = registry.compare_versions("face_detector", "1.0.0", "1.1.0")
        print("Version comparison:")
        print(json.dumps(comparison, indent=2))
        print()
        
        # Get lineage
        lineage = registry.get_lineage("face_detector", "1.1.0")
        print("Model lineage for v1.1.0:")
        for i, model in enumerate(lineage):
            print(f"  {i}. v{model.version} (accuracy: {model.accuracy})")
        print()
        
        # Cleanup
        dummy_model.unlink()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
