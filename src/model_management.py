# Advanced Model Management System

import os
import json
import pickle
import hashlib
import shutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import logging
from abc import ABC, abstractmethod
import threading
from collections import defaultdict

@dataclass
class ModelMetadata:
    """Model metadata structure."""
    name: str
    version: str
    model_type: str
    created_at: datetime
    file_path: str
    file_size: int
    checksum: str
    performance_metrics: Dict[str, float]
    training_config: Dict[str, Any]
    dependencies: List[str]
    description: str
    tags: List[str]
    is_active: bool
    last_used: Optional[datetime] = None
    usage_count: int = 0

@dataclass
class ModelPerformance:
    """Model performance tracking."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_avg: float
    memory_usage: float
    throughput: float
    last_evaluated: datetime

class ModelVersionControl:
    """Version control system for machine learning models."""
    
    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.models_registry = {}
        self.active_models = {}
        self.performance_history = defaultdict(list)
        
        # Configuration
        self.max_versions_per_model = 10
        self.auto_cleanup_enabled = True
        self.backup_enabled = True
        
        # Load existing models
        self._load_registry()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _load_registry(self):
        """Load models registry from disk."""
        registry_file = self.base_path / "registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to ModelMetadata objects
                for model_data in data.get("models", []):
                    metadata = ModelMetadata(**{
                        **model_data,
                        "created_at": datetime.fromisoformat(model_data["created_at"]),
                        "last_used": datetime.fromisoformat(model_data["last_used"]) if model_data.get("last_used") else None
                    })
                    
                    key = f"{metadata.name}:{metadata.version}"
                    self.models_registry[key] = metadata
                    
                    if metadata.is_active:
                        self.active_models[metadata.name] = metadata
                        
            except Exception as e:
                logging.error(f"Failed to load models registry: {e}")
    
    def _save_registry(self):
        """Save models registry to disk."""
        registry_file = self.base_path / "registry.json"
        
        try:
            # Convert to serializable format
            registry_data = {
                "models": [],
                "last_updated": datetime.now().isoformat()
            }
            
            for metadata in self.models_registry.values():
                model_data = asdict(metadata)
                model_data["created_at"] = metadata.created_at.isoformat()
                model_data["last_used"] = metadata.last_used.isoformat() if metadata.last_used else None
                registry_data["models"].append(model_data)
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save models registry: {e}")
    
    def register_model(self, model_obj: Any, name: str, version: str, 
                      model_type: str, description: str = "",
                      tags: List[str] = None, 
                      training_config: Dict = None,
                      performance_metrics: Dict = None) -> str:
        """Register a new model version."""
        
        if tags is None:
            tags = []
        if training_config is None:
            training_config = {}
        if performance_metrics is None:
            performance_metrics = {}
        
        # Create model directory
        model_dir = self.base_path / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_dir / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_obj, f)
        
        # Calculate checksum
        checksum = self._calculate_checksum(model_file)
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            model_type=model_type,
            created_at=datetime.now(),
            file_path=str(model_file),
            file_size=model_file.stat().st_size,
            checksum=checksum,
            performance_metrics=performance_metrics,
            training_config=training_config,
            dependencies=self._get_dependencies(),
            description=description,
            tags=tags,
            is_active=False
        )
        
        # Register model
        key = f"{name}:{version}"
        self.models_registry[key] = metadata
        
        # Save training config separately
        config_file = model_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(training_config, f, indent=2)
        
        # Create backup if enabled
        if self.backup_enabled:
            self._create_backup(metadata)
        
        # Cleanup old versions if needed
        if self.auto_cleanup_enabled:
            self._cleanup_old_versions(name)
        
        # Save registry
        self._save_registry()
        
        logging.info(f"Registered model {name}:{version}")
        return key
    
    def load_model(self, name: str, version: str = None) -> Tuple[Any, ModelMetadata]:
        """Load a specific model version."""
        
        if version is None:
            # Load active version
            if name in self.active_models:
                metadata = self.active_models[name]
            else:
                # Load latest version
                versions = self.get_model_versions(name)
                if not versions:
                    raise ValueError(f"No models found for {name}")
                metadata = versions[0]  # Latest version
        else:
            key = f"{name}:{version}"
            if key not in self.models_registry:
                raise ValueError(f"Model {key} not found")
            metadata = self.models_registry[key]
        
        # Verify file exists
        if not os.path.exists(metadata.file_path):
            raise FileNotFoundError(f"Model file not found: {metadata.file_path}")
        
        # Verify checksum
        current_checksum = self._calculate_checksum(metadata.file_path)
        if current_checksum != metadata.checksum:
            raise ValueError(f"Model file corrupted: checksum mismatch")
        
        # Load model
        with open(metadata.file_path, 'rb') as f:
            model = pickle.load(f)
        
        # Update usage statistics
        metadata.last_used = datetime.now()
        metadata.usage_count += 1
        self._save_registry()
        
        logging.info(f"Loaded model {metadata.name}:{metadata.version}")
        return model, metadata
    
    def set_active_model(self, name: str, version: str) -> bool:
        """Set a specific version as the active model."""
        key = f"{name}:{version}"
        
        if key not in self.models_registry:
            raise ValueError(f"Model {key} not found")
        
        # Deactivate current active model
        if name in self.active_models:
            old_active = self.active_models[name]
            old_active.is_active = False
        
        # Activate new model
        new_active = self.models_registry[key]
        new_active.is_active = True
        self.active_models[name] = new_active
        
        self._save_registry()
        
        logging.info(f"Set {key} as active model")
        return True
    
    def get_model_versions(self, name: str) -> List[ModelMetadata]:
        """Get all versions of a model, sorted by creation date (newest first)."""
        versions = [
            metadata for key, metadata in self.models_registry.items()
            if metadata.name == name
        ]
        return sorted(versions, key=lambda x: x.created_at, reverse=True)
    
    def compare_models(self, name: str, versions: List[str] = None) -> Dict:
        """Compare performance metrics across model versions."""
        if versions is None:
            model_versions = self.get_model_versions(name)
            versions = [m.version for m in model_versions[:5]]  # Compare latest 5
        
        comparison = {
            "model_name": name,
            "versions": {},
            "metrics_comparison": defaultdict(dict)
        }
        
        for version in versions:
            key = f"{name}:{version}"
            if key in self.models_registry:
                metadata = self.models_registry[key]
                comparison["versions"][version] = {
                    "created_at": metadata.created_at.isoformat(),
                    "performance_metrics": metadata.performance_metrics,
                    "usage_count": metadata.usage_count,
                    "is_active": metadata.is_active
                }
                
                # Organize metrics for comparison
                for metric, value in metadata.performance_metrics.items():
                    comparison["metrics_comparison"][metric][version] = value
        
        return comparison
    
    def update_performance_metrics(self, name: str, version: str, 
                                 performance: ModelPerformance):
        """Update performance metrics for a model."""
        key = f"{name}:{version}"
        
        if key not in self.models_registry:
            raise ValueError(f"Model {key} not found")
        
        metadata = self.models_registry[key]
        metadata.performance_metrics.update(asdict(performance))
        
        # Store in performance history
        self.performance_history[key].append({
            "timestamp": datetime.now().isoformat(),
            "performance": asdict(performance)
        })
        
        self._save_registry()
        
        logging.info(f"Updated performance metrics for {key}")
    
    def get_model_info(self, name: str, version: str = None) -> Dict:
        """Get comprehensive information about a model."""
        if version is None:
            if name in self.active_models:
                metadata = self.active_models[name]
            else:
                raise ValueError(f"No active model for {name}")
        else:
            key = f"{name}:{version}"
            if key not in self.models_registry:
                raise ValueError(f"Model {key} not found")
            metadata = self.models_registry[key]
        
        # Get performance history
        key = f"{metadata.name}:{metadata.version}"
        performance_history = self.performance_history.get(key, [])
        
        return {
            "metadata": asdict(metadata),
            "performance_history": performance_history,
            "file_exists": os.path.exists(metadata.file_path),
            "model_size_mb": metadata.file_size / (1024 * 1024),
            "age_days": (datetime.now() - metadata.created_at).days
        }
    
    def delete_model(self, name: str, version: str, force: bool = False) -> bool:
        """Delete a specific model version."""
        key = f"{name}:{version}"
        
        if key not in self.models_registry:
            raise ValueError(f"Model {key} not found")
        
        metadata = self.models_registry[key]
        
        # Check if it's the active model
        if metadata.is_active and not force:
            raise ValueError("Cannot delete active model. Use force=True or set another version as active first.")
        
        # Remove files
        model_dir = Path(metadata.file_path).parent
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del self.models_registry[key]
        
        # Remove from active models if needed
        if name in self.active_models and self.active_models[name].version == version:
            del self.active_models[name]
        
        # Remove from performance history
        if key in self.performance_history:
            del self.performance_history[key]
        
        self._save_registry()
        
        logging.info(f"Deleted model {key}")
        return True
    
    def export_model(self, name: str, version: str, export_path: str) -> str:
        """Export a model package for deployment."""
        key = f"{name}:{version}"
        
        if key not in self.models_registry:
            raise ValueError(f"Model {key} not found")
        
        metadata = self.models_registry[key]
        
        # Create export directory
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create package structure
        package_dir = export_dir / f"{name}_{version}"
        package_dir.mkdir(exist_ok=True)
        
        # Copy model file
        model_source = Path(metadata.file_path)
        model_dest = package_dir / "model.pkl"
        shutil.copy2(model_source, model_dest)
        
        # Create deployment metadata
        deployment_metadata = {
            "model_name": name,
            "model_version": version,
            "model_type": metadata.model_type,
            "created_at": metadata.created_at.isoformat(),
            "performance_metrics": metadata.performance_metrics,
            "dependencies": metadata.dependencies,
            "deployment_created": datetime.now().isoformat()
        }
        
        with open(package_dir / "deployment_metadata.json", 'w') as f:
            json.dump(deployment_metadata, f, indent=2)
        
        # Copy training config
        config_source = model_source.parent / "training_config.json"
        if config_source.exists():
            shutil.copy2(config_source, package_dir / "training_config.json")
        
        # Create deployment script
        deployment_script = f"""#!/usr/bin/env python3
'''
Auto-generated deployment script for {name}:{version}
Generated at: {datetime.now().isoformat()}
'''

import pickle
import json
from pathlib import Path

class ModelDeployment:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = Path(__file__).parent
        else:
            model_dir = Path(model_dir)
        
        self.model_dir = model_dir
        self.model = None
        self.metadata = None
        
        self._load_model()
    
    def _load_model(self):
        # Load model
        model_file = self.model_dir / "model.pkl"
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metadata
        metadata_file = self.model_dir / "deployment_metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded model {{self.metadata['model_name']}}:{{self.metadata['model_version']}}")
    
    def predict(self, *args, **kwargs):
        if hasattr(self.model, 'predict'):
            return self.model.predict(*args, **kwargs)
        elif callable(self.model):
            return self.model(*args, **kwargs)
        else:
            raise NotImplementedError("Model does not have predict method or is not callable")
    
    def get_info(self):
        return self.metadata

if __name__ == "__main__":
    deployment = ModelDeployment()
    print("Model deployment ready!")
    print(f"Model info: {{deployment.get_info()}}")
"""
        
        with open(package_dir / "deploy.py", 'w') as f:
            f.write(deployment_script)
        
        # Make deployment script executable
        os.chmod(package_dir / "deploy.py", 0o755)
        
        logging.info(f"Exported model {key} to {package_dir}")
        return str(package_dir)
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_dependencies(self) -> List[str]:
        """Get current Python dependencies."""
        try:
            import pkg_resources
            installed_packages = [d for d in pkg_resources.working_set]
            return [f"{p.project_name}=={p.version}" for p in installed_packages]
        except Exception:
            return []
    
    def _create_backup(self, metadata: ModelMetadata):
        """Create backup of model."""
        backup_dir = self.base_path / "backups" / metadata.name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_file = backup_dir / f"{metadata.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        shutil.copy2(metadata.file_path, backup_file)
    
    def _cleanup_old_versions(self, model_name: str):
        """Clean up old model versions."""
        versions = self.get_model_versions(model_name)
        
        if len(versions) > self.max_versions_per_model:
            # Keep the most recent versions and active version
            to_keep = set()
            
            # Keep latest versions
            for version in versions[:self.max_versions_per_model-1]:
                to_keep.add(version.version)
            
            # Keep active version
            if model_name in self.active_models:
                to_keep.add(self.active_models[model_name].version)
            
            # Delete old versions
            for version in versions:
                if version.version not in to_keep:
                    try:
                        self.delete_model(model_name, version.version, force=True)
                        logging.info(f"Cleaned up old version {model_name}:{version.version}")
                    except Exception as e:
                        logging.error(f"Failed to cleanup {model_name}:{version.version}: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        def background_maintenance():
            while True:
                try:
                    # Periodic registry backup
                    time.sleep(3600)  # Every hour
                    self._save_registry()
                    
                    # Performance monitoring
                    # This could include checking model drift, performance degradation, etc.
                    
                except Exception as e:
                    logging.error(f"Background maintenance error: {e}")
        
        maintenance_thread = threading.Thread(target=background_maintenance, daemon=True)
        maintenance_thread.start()
    
    def get_system_stats(self) -> Dict:
        """Get model management system statistics."""
        total_models = len(self.models_registry)
        active_models_count = len(self.active_models)
        
        # Calculate total storage used
        total_size = sum(metadata.file_size for metadata in self.models_registry.values())
        
        # Model types distribution
        model_types = defaultdict(int)
        for metadata in self.models_registry.values():
            model_types[metadata.model_type] += 1
        
        # Usage statistics
        total_usage = sum(metadata.usage_count for metadata in self.models_registry.values())
        
        return {
            "total_models": total_models,
            "active_models": active_models_count,
            "total_storage_mb": total_size / (1024 * 1024),
            "model_types": dict(model_types),
            "total_usage_count": total_usage,
            "registry_file": str(self.base_path / "registry.json"),
            "auto_cleanup_enabled": self.auto_cleanup_enabled,
            "max_versions_per_model": self.max_versions_per_model
        }


# Model performance monitoring
class ModelMonitor:
    """Monitor model performance and detect drift."""
    
    def __init__(self, model_manager: ModelVersionControl):
        self.model_manager = model_manager
        self.performance_thresholds = {
            "accuracy_drop": 0.05,  # 5% drop
            "response_time_increase": 2.0,  # 2x increase
            "error_rate_increase": 0.02  # 2% increase
        }
        self.monitoring_enabled = True
    
    def evaluate_model_performance(self, name: str, version: str, 
                                 test_data: Any, test_labels: Any) -> ModelPerformance:
        """Evaluate model performance on test data."""
        model, metadata = self.model_manager.load_model(name, version)
        
        start_time = time.time()
        
        # Make predictions
        predictions = model.predict(test_data)
        
        inference_time = time.time() - start_time
        
        # Calculate metrics (this is a simplified example)
        # In practice, you'd use proper evaluation metrics
        accuracy = np.mean(predictions == test_labels) if hasattr(predictions, '__len__') else 0.0
        
        performance = ModelPerformance(
            accuracy=accuracy,
            precision=accuracy,  # Simplified
            recall=accuracy,     # Simplified
            f1_score=accuracy,   # Simplified
            inference_time_avg=inference_time / len(test_data),
            memory_usage=0.0,    # Would measure actual memory usage
            throughput=len(test_data) / inference_time,
            last_evaluated=datetime.now()
        )
        
        # Update performance metrics
        self.model_manager.update_performance_metrics(name, version, performance)
        
        return performance
    
    def detect_performance_drift(self, name: str, current_performance: ModelPerformance) -> Dict:
        """Detect performance drift compared to baseline."""
        versions = self.model_manager.get_model_versions(name)
        
        if len(versions) < 2:
            return {"drift_detected": False, "reason": "Insufficient historical data"}
        
        # Use first version as baseline
        baseline_metadata = versions[-1]  # Oldest version
        baseline_metrics = baseline_metadata.performance_metrics
        
        drift_alerts = []
        
        # Check accuracy drift
        if "accuracy" in baseline_metrics:
            accuracy_drop = baseline_metrics["accuracy"] - current_performance.accuracy
            if accuracy_drop > self.performance_thresholds["accuracy_drop"]:
                drift_alerts.append(f"Accuracy dropped by {accuracy_drop:.2%}")
        
        # Check response time drift
        if "inference_time_avg" in baseline_metrics:
            time_increase = current_performance.inference_time_avg / baseline_metrics["inference_time_avg"]
            if time_increase > self.performance_thresholds["response_time_increase"]:
                drift_alerts.append(f"Response time increased by {time_increase:.1f}x")
        
        return {
            "drift_detected": len(drift_alerts) > 0,
            "alerts": drift_alerts,
            "baseline_version": baseline_metadata.version,
            "current_performance": asdict(current_performance)
        }


# Example usage
if __name__ == "__main__":
    # Initialize model manager
    model_manager = ModelVersionControl("./models")
    
    # Example: Register a dummy model
    class DummyModel:
        def predict(self, X):
            return np.random.rand(len(X))
    
    dummy_model = DummyModel()
    
    # Register model
    model_id = model_manager.register_model(
        model_obj=dummy_model,
        name="face_classifier",
        version="1.0.0",
        model_type="classification",
        description="Initial face classification model",
        tags=["face", "classification", "v1"],
        training_config={"epochs": 100, "batch_size": 32},
        performance_metrics={"accuracy": 0.95, "f1_score": 0.93}
    )
    
    print(f"Registered model: {model_id}")
    
    # Set as active
    model_manager.set_active_model("face_classifier", "1.0.0")
    
    # Load model
    loaded_model, metadata = model_manager.load_model("face_classifier")
    print(f"Loaded model: {metadata.name}:{metadata.version}")
    
    # Get system stats
    stats = model_manager.get_system_stats()
    print(f"System stats: {stats}")
    
    # Export model
    export_path = model_manager.export_model("face_classifier", "1.0.0", "./exports")
    print(f"Exported to: {export_path}")