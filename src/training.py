"""
Training pipeline for custom face recognition models.
Supports fine-tuning existing models and training from scratch.
"""

import os
import json
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pickle
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

# ML and deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import cv2
    from PIL import Image
except ImportError:
    # Create mock classes for development
    torch = None
    nn = None
    optim = None
    Dataset = object
    DataLoader = None
    transforms = None

# Import face recognition components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from database import DatabaseManager
    from preprocessing import FacePreprocessor
    from detection import FaceDetector
    from monitoring import performance_monitor
except ImportError:
    # Mock for development
    DatabaseManager = object
    FacePreprocessor = object
    FaceDetector = object
    performance_monitor = object

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    model_name: str = "custom_face_recognizer"
    model_type: str = "siamese"  # siamese, triplet, classification
    input_size: Tuple[int, int] = (112, 112)
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    save_best_only: bool = True
    output_dir: str = "models/training"
    data_augmentation: bool = True
    pretrained_backbone: str = "resnet50"
    freeze_backbone: bool = False
    embedding_dim: int = 512
    margin: float = 0.5  # For triplet loss
    min_samples_per_person: int = 5
    max_samples_per_person: int = 100

class FaceRecognitionDataset(Dataset):
    """
    Dataset class for face recognition training.
    """
    
    def __init__(
        self,
        face_data: List[Dict],
        transform=None,
        mode: str = "classification"
    ):
        """
        Initialize dataset.
        
        Args:
            face_data: List of face data dictionaries
            transform: Image transformations
            mode: Training mode (classification, siamese, triplet)
        """
        self.face_data = face_data
        self.transform = transform
        self.mode = mode
        
        # Create person to ID mapping
        self.person_to_id = {}
        person_names = list(set([item['person_name'] for item in face_data]))
        for i, name in enumerate(sorted(person_names)):
            self.person_to_id[name] = i
        
        self.num_classes = len(self.person_to_id)
        
        # Group data by person for triplet/siamese modes
        self.person_groups = {}
        for item in face_data:
            person_name = item['person_name']
            if person_name not in self.person_groups:
                self.person_groups[person_name] = []
            self.person_groups[person_name].append(item)
    
    def __len__(self):
        if self.mode == "triplet":
            return len(self.face_data) * 2  # Generate more triplets
        return len(self.face_data)
    
    def __getitem__(self, idx):
        if self.mode == "classification":
            return self._get_classification_item(idx)
        elif self.mode == "siamese":
            return self._get_siamese_item(idx)
        elif self.mode == "triplet":
            return self._get_triplet_item(idx)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _get_classification_item(self, idx):
        """Get item for classification training."""
        item = self.face_data[idx]
        
        # Load and process image
        image = self._load_image(item['image_path'])
        if self.transform:
            image = self.transform(image)
        
        label = self.person_to_id[item['person_name']]
        
        return image, label
    
    def _get_siamese_item(self, idx):
        """Get item for Siamese network training."""
        # Randomly decide if positive or negative pair
        is_positive = np.random.choice([True, False])
        
        item1 = self.face_data[idx % len(self.face_data)]
        person1 = item1['person_name']
        
        if is_positive:
            # Same person
            candidates = self.person_groups[person1]
            if len(candidates) > 1:
                item2 = np.random.choice([c for c in candidates if c != item1])
            else:
                item2 = item1
            label = 1
        else:
            # Different person
            other_persons = [p for p in self.person_groups.keys() if p != person1]
            if other_persons:
                other_person = np.random.choice(other_persons)
                item2 = np.random.choice(self.person_groups[other_person])
            else:
                item2 = item1
            label = 0
        
        image1 = self._load_image(item1['image_path'])
        image2 = self._load_image(item2['image_path'])
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        return (image1, image2), label
    
    def _get_triplet_item(self, idx):
        """Get item for triplet loss training."""
        # Anchor
        anchor_item = self.face_data[idx % len(self.face_data)]
        anchor_person = anchor_item['person_name']
        
        # Positive (same person)
        positive_candidates = [c for c in self.person_groups[anchor_person] if c != anchor_item]
        if positive_candidates:
            positive_item = np.random.choice(positive_candidates)
        else:
            positive_item = anchor_item
        
        # Negative (different person)
        other_persons = [p for p in self.person_groups.keys() if p != anchor_person]
        if other_persons:
            negative_person = np.random.choice(other_persons)
            negative_item = np.random.choice(self.person_groups[negative_person])
        else:
            # Fallback to random item
            negative_item = np.random.choice(self.face_data)
        
        anchor_image = self._load_image(anchor_item['image_path'])
        positive_image = self._load_image(positive_item['image_path'])
        negative_image = self._load_image(negative_item['image_path'])
        
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)
        
        return (anchor_image, positive_image, negative_image), 0  # Label not used in triplet
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from path."""
        if isinstance(image_path, str) and os.path.exists(image_path):
            return Image.open(image_path).convert('RGB')
        else:
            # Create dummy image for development
            return Image.new('RGB', (112, 112), color='gray')

class FaceRecognitionModel(nn.Module):
    """
    Face recognition model with configurable architecture.
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        backbone: str = "resnet50",
        mode: str = "classification"
    ):
        super(FaceRecognitionModel, self).__init__()
        
        self.mode = mode
        self.embedding_dim = embedding_dim
        
        # Backbone network
        if torch:
            if backbone == "resnet50":
                self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
                self.backbone.fc = nn.Identity()  # Remove final layer
                backbone_out_features = 2048
            else:
                # Simple CNN backbone as fallback
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d((7, 7)),
                    nn.Flatten(),
                    nn.Linear(128 * 7 * 7, 512)
                )
                backbone_out_features = 512
        else:
            backbone_out_features = 512
            self.backbone = None
        
        # Embedding layer
        if torch:
            self.embedding = nn.Sequential(
                nn.Linear(backbone_out_features, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU()
            )
        
        # Classification head (only for classification mode)
        if mode == "classification" and torch:
            self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        if not torch:
            return None
        
        if isinstance(x, tuple):
            # Multiple inputs (for Siamese/triplet)
            embeddings = []
            for input_tensor in x:
                features = self.backbone(input_tensor)
                embedding = self.embedding(features)
                embeddings.append(embedding)
            return embeddings
        else:
            # Single input
            features = self.backbone(x)
            embedding = self.embedding(features)
            
            if self.mode == "classification":
                logits = self.classifier(embedding)
                return embedding, logits
            else:
                return embedding

class ModelTrainer:
    """
    Main training pipeline for face recognition models.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.face_detector = FaceDetector()
        self.preprocessor = FacePreprocessor()
        
        # Training metrics
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        logger.info(f"Trainer initialized with device: {self.device}")
    
    def prepare_dataset(self) -> Tuple[List[Dict], List[str]]:
        """
        Prepare training dataset from database.
        
        Returns:
            Tuple of (face_data, person_names)
        """
        logger.info("Preparing dataset from database...")
        
        # Get all persons with minimum sample count
        persons = self.db_manager.list_persons()
        valid_persons = []
        
        for person in persons:
            embeddings = self.db_manager.get_person_embeddings(person.id)
            if len(embeddings) >= self.config.min_samples_per_person:
                valid_persons.append(person)
        
        logger.info(f"Found {len(valid_persons)} persons with sufficient samples")
        
        # Collect face data
        face_data = []
        person_names = []
        
        for person in valid_persons:
            person_names.append(person.name)
            embeddings = self.db_manager.get_person_embeddings(person.id)
            
            # Limit samples per person
            if len(embeddings) > self.config.max_samples_per_person:
                embeddings = np.random.choice(
                    embeddings, 
                    self.config.max_samples_per_person, 
                    replace=False
                )
            
            for embedding in embeddings:
                face_data.append({
                    'person_id': person.id,
                    'person_name': person.name,
                    'embedding_id': embedding.id,
                    'image_path': embedding.source_image_path,
                    'quality_score': embedding.quality_score
                })
        
        logger.info(f"Prepared dataset with {len(face_data)} samples from {len(person_names)} persons")
        
        return face_data, person_names
    
    def create_data_loaders(self, face_data: List[Dict]) -> Tuple:
        """
        Create training and validation data loaders.
        
        Args:
            face_data: List of face data dictionaries
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Split data
        train_data, val_data = train_test_split(
            face_data,
            test_size=self.config.validation_split,
            stratify=[item['person_name'] for item in face_data],
            random_state=42
        )
        
        # Data augmentation transforms
        if self.config.data_augmentation and transforms:
            train_transform = transforms.Compose([
                transforms.Resize(self.config.input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = None
        
        # Validation transforms
        if transforms:
            val_transform = transforms.Compose([
                transforms.Resize(self.config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            val_transform = None
        
        # Create datasets
        train_dataset = FaceRecognitionDataset(
            train_data, 
            transform=train_transform, 
            mode=self.config.model_type
        )
        val_dataset = FaceRecognitionDataset(
            val_data, 
            transform=val_transform, 
            mode=self.config.model_type
        )
        
        if DataLoader:
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        else:
            train_loader = val_loader = None
        
        logger.info(f"Created data loaders: train={len(train_data)}, val={len(val_data)}")
        
        return train_loader, val_loader, train_dataset.num_classes
    
    def create_model(self, num_classes: int) -> nn.Module:
        """
        Create and initialize model.
        
        Args:
            num_classes: Number of classes for classification
            
        Returns:
            Initialized model
        """
        if not torch:
            logger.warning("PyTorch not available, returning None model")
            return None
        
        model = FaceRecognitionModel(
            num_classes=num_classes,
            embedding_dim=self.config.embedding_dim,
            backbone=self.config.pretrained_backbone,
            mode=self.config.model_type
        )
        
        # Freeze backbone if requested
        if self.config.freeze_backbone and hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created: {total_params:,} total params, {trainable_params:,} trainable")
        
        return model
    
    def get_loss_function(self):
        """Get appropriate loss function for model type."""
        if not torch:
            return None
        
        if self.config.model_type == "classification":
            return nn.CrossEntropyLoss()
        elif self.config.model_type == "siamese":
            return nn.BCEWithLogitsLoss()
        elif self.config.model_type == "triplet":
            return nn.TripletMarginLoss(margin=self.config.margin)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch."""
        if not torch:
            return 0.0, 0.0
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            if self.config.model_type == "classification":
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                embeddings, logits = model(data)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            # Add other training modes here (siamese, triplet)
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                    f'Loss: {loss.item():.4f}'
                )
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch."""
        if not torch:
            return 0.0, 0.0
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                if self.config.model_type == "classification":
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                    
                    embeddings, logits = model(data)
                    loss = criterion(logits, targets)
                    
                    _, predicted = torch.max(logits.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    @performance_monitor.time_function("model_training")
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training results dictionary
        """
        logger.info("Starting model training...")
        
        # Prepare dataset
        face_data, person_names = self.prepare_dataset()
        if len(face_data) == 0:
            raise ValueError("No training data available")
        
        # Create data loaders
        train_loader, val_loader, num_classes = self.create_data_loaders(face_data)
        
        # Create model
        model = self.create_model(num_classes)
        if model is None:
            logger.warning("Model creation failed, returning mock results")
            return {"status": "failed", "reason": "PyTorch not available"}
        
        # Loss function and optimizer
        criterion = self.get_loss_function()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # Training variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )
            
            # Validate
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rate'].append(current_lr)
            
            logger.info(
                f'Epoch {epoch+1}/{self.config.num_epochs} - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - '
                f'LR: {current_lr:.6f}'
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if self.config.save_best_only:
                    self.save_model(model, f"best_model_epoch_{epoch+1}")
                    logger.info(f"Saved best model at epoch {epoch+1}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save final model and training history
        self.save_model(model, "final_model")
        self.save_training_history()
        
        # Generate training report
        results = self.generate_training_report()
        
        logger.info("Training completed successfully")
        return results
    
    def save_model(self, model, name: str):
        """Save model to disk."""
        if not torch or model is None:
            return
        
        model_path = self.output_dir / f"{name}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def save_training_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        if not self.training_history['train_loss']:
            return {"status": "no_data"}
        
        best_epoch = np.argmin(self.training_history['val_loss'])
        
        report = {
            "status": "completed",
            "config": asdict(self.config),
            "training_summary": {
                "total_epochs": len(self.training_history['train_loss']),
                "best_epoch": best_epoch + 1,
                "best_val_loss": self.training_history['val_loss'][best_epoch],
                "best_val_acc": self.training_history['val_acc'][best_epoch],
                "final_train_loss": self.training_history['train_loss'][-1],
                "final_val_loss": self.training_history['val_loss'][-1],
                "final_train_acc": self.training_history['train_acc'][-1],
                "final_val_acc": self.training_history['val_acc'][-1]
            },
            "model_info": {
                "embedding_dim": self.config.embedding_dim,
                "backbone": self.config.pretrained_backbone,
                "model_type": self.config.model_type
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def train_custom_model(config_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Train a custom face recognition model.
    
    Args:
        config_dict: Training configuration dictionary
        
    Returns:
        Training results
    """
    # Create config
    if config_dict:
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Start training
    results = trainer.train()
    
    return results

if __name__ == "__main__":
    # Example usage
    config = {
        "model_name": "custom_face_model",
        "model_type": "classification",
        "num_epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001
    }
    
    results = train_custom_model(config)
    print(f"Training completed: {results['status']}")