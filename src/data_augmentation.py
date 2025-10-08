"""
Data augmentation and dataset management for face recognition training.
Provides advanced augmentation techniques and dataset organization tools.
"""

import os
import json
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import shutil
import random
from dataclasses import dataclass
import hashlib

# Image processing imports
try:
    from PIL import Image, ImageEnhance, ImageFilter
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import imgaug.augmenters as iaa
except ImportError:
    # Mock for development
    Image = None
    ImageEnhance = None
    ImageFilter = None
    A = None
    ToTensorV2 = None
    iaa = None

# Import face recognition components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from database import DatabaseManager
    from detection import FaceDetector
    from preprocessing import FacePreprocessor
    from monitoring import performance_monitor
except ImportError:
    # Mock for development
    DatabaseManager = object
    FaceDetector = object
    FacePreprocessor = object
    performance_monitor = object

logger = logging.getLogger(__name__)

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # Geometric transformations
    rotation_range: Tuple[int, int] = (-15, 15)
    zoom_range: Tuple[float, float] = (0.9, 1.1)
    shift_range: Tuple[float, float] = (-0.1, 0.1)
    horizontal_flip: bool = True
    vertical_flip: bool = False
    
    # Appearance transformations
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_range: Tuple[int, int] = (-10, 10)
    
    # Noise and blur
    gaussian_noise_var: Tuple[float, float] = (0, 0.01)
    gaussian_blur_sigma: Tuple[float, float] = (0, 1.0)
    motion_blur_kernel: Tuple[int, int] = (3, 7)
    
    # Advanced augmentations
    cutout_holes: int = 2
    cutout_size: Tuple[int, int] = (8, 16)
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Quality control
    preserve_quality_threshold: float = 0.5
    max_augmentations_per_image: int = 5

class DataAugmenter:
    """
    Advanced data augmentation system for face images.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        """
        Initialize augmenter.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()
        self.face_detector = FaceDetector()
        self.preprocessor = FacePreprocessor()
        
        # Initialize augmentation pipelines
        self._setup_augmentation_pipelines()
        
    def _setup_augmentation_pipelines(self):
        """Setup different augmentation pipelines."""
        
        # Basic augmentations
        if A:
            self.basic_augmentations = A.Compose([
                A.Rotate(limit=self.config.rotation_range, p=0.5),
                A.RandomScale(scale_limit=0.1, p=0.3),
                A.HorizontalFlip(p=0.5 if self.config.horizontal_flip else 0),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.3
                )
            ])
            
            # Color augmentations
            self.color_augmentations = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3)
            ])
            
            # Noise and blur augmentations
            self.noise_augmentations = A.Compose([
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.MotionBlur(blur_limit=5, p=0.2)
            ])
            
            # Advanced augmentations
            self.advanced_augmentations = A.Compose([
                A.CoarseDropout(
                    max_holes=self.config.cutout_holes,
                    max_height=self.config.cutout_size[1],
                    max_width=self.config.cutout_size[1],
                    p=0.3
                ),
                A.GridDistortion(p=0.2),
                A.ElasticTransform(p=0.2),
                A.OpticalDistortion(distort_limit=0.1, p=0.2)
            ])
        else:
            # Mock pipelines for development
            self.basic_augmentations = None
            self.color_augmentations = None
            self.noise_augmentations = None
            self.advanced_augmentations = None
        
        # ImgAug pipeline for complex augmentations
        if iaa:
            self.imgaug_pipeline = iaa.Sequential([
                iaa.Sometimes(0.3, iaa.Affine(
                    rotate=(-15, 15),
                    scale=(0.9, 1.1),
                    translate_percent=(-0.1, 0.1)
                )),
                iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.01, 0.05))),
                iaa.Sometimes(0.2, iaa.PiecewiseAffine(scale=(0.01, 0.03))),
                iaa.Sometimes(0.2, iaa.LinearContrast((0.8, 1.2))),
                iaa.Sometimes(0.2, iaa.Multiply((0.8, 1.2)))
            ])
        else:
            self.imgaug_pipeline = None
    
    def augment_image(
        self,
        image: np.ndarray,
        augmentation_type: str = "basic",
        preserve_faces: bool = True
    ) -> List[np.ndarray]:
        """
        Apply augmentations to a single image.
        
        Args:
            image: Input image as numpy array
            augmentation_type: Type of augmentation ('basic', 'color', 'noise', 'advanced', 'all')
            preserve_faces: Whether to ensure faces are preserved after augmentation
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        # Original image
        augmented_images.append(image.copy())
        
        # Apply different augmentation pipelines
        if augmentation_type == "basic" or augmentation_type == "all":
            if self.basic_augmentations:
                for _ in range(2):
                    aug_image = self.basic_augmentations(image=image)["image"]
                    if self._validate_augmented_image(aug_image, image, preserve_faces):
                        augmented_images.append(aug_image)
        
        if augmentation_type == "color" or augmentation_type == "all":
            if self.color_augmentations:
                for _ in range(2):
                    aug_image = self.color_augmentations(image=image)["image"]
                    if self._validate_augmented_image(aug_image, image, preserve_faces):
                        augmented_images.append(aug_image)
        
        if augmentation_type == "noise" or augmentation_type == "all":
            if self.noise_augmentations:
                aug_image = self.noise_augmentations(image=image)["image"]
                if self._validate_augmented_image(aug_image, image, preserve_faces):
                    augmented_images.append(aug_image)
        
        if augmentation_type == "advanced" or augmentation_type == "all":
            if self.advanced_augmentations:
                aug_image = self.advanced_augmentations(image=image)["image"]
                if self._validate_augmented_image(aug_image, image, preserve_faces):
                    augmented_images.append(aug_image)
        
        # ImgAug augmentations
        if self.imgaug_pipeline and (augmentation_type == "advanced" or augmentation_type == "all"):
            aug_image = self.imgaug_pipeline(image=image)
            if self._validate_augmented_image(aug_image, image, preserve_faces):
                augmented_images.append(aug_image)
        
        # Limit number of augmentations
        if len(augmented_images) > self.config.max_augmentations_per_image:
            augmented_images = augmented_images[:self.config.max_augmentations_per_image]
        
        return augmented_images
    
    def _validate_augmented_image(
        self,
        augmented_image: np.ndarray,
        original_image: np.ndarray,
        preserve_faces: bool
    ) -> bool:
        """
        Validate that augmented image meets quality criteria.
        
        Args:
            augmented_image: Augmented image
            original_image: Original image
            preserve_faces: Whether to check for face preservation
            
        Returns:
            True if image passes validation
        """
        if not preserve_faces:
            return True
        
        try:
            # Check if faces are still detectable
            original_faces = self.face_detector.detect_faces(original_image)
            augmented_faces = self.face_detector.detect_faces(augmented_image)
            
            # Must have at least one face in both
            if len(original_faces) == 0 or len(augmented_faces) == 0:
                return False
            
            # Check image quality (simple metric)
            quality_score = self.preprocessor.assess_image_quality(augmented_image)
            if quality_score < self.config.preserve_quality_threshold:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating augmented image: {str(e)}")
            return False
    
    def mixup_images(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        alpha: float = None
    ) -> Tuple[np.ndarray, float]:
        """
        Apply MixUp augmentation to two images.
        
        Args:
            image1: First image
            image2: Second image
            alpha: MixUp parameter
            
        Returns:
            Mixed image and lambda value
        """
        if alpha is None:
            alpha = self.config.mixup_alpha
        
        # Sample lambda from beta distribution
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        # Ensure images have same shape
        if image1.shape != image2.shape:
            # Resize image2 to match image1
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # Mix images
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_image = np.clip(mixed_image, 0, 255).astype(np.uint8)
        
        return mixed_image, lam
    
    def cutmix_images(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        alpha: float = None
    ) -> Tuple[np.ndarray, float]:
        """
        Apply CutMix augmentation to two images.
        
        Args:
            image1: First image
            image2: Second image
            alpha: CutMix parameter
            
        Returns:
            Mixed image and lambda value
        """
        if alpha is None:
            alpha = self.config.cutmix_alpha
        
        h, w = image1.shape[:2]
        
        # Sample lambda from beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Sample bounding box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Ensure image2 has same shape
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (w, h))
        
        # Apply cutmix
        mixed_image = image1.copy()
        mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        return mixed_image, lam

class DatasetManager:
    """
    Comprehensive dataset management for face recognition training.
    """
    
    def __init__(self, dataset_root: str):
        """
        Initialize dataset manager.
        
        Args:
            dataset_root: Root directory for datasets
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.augmenter = DataAugmenter()
        self.face_detector = FaceDetector()
        
        # Dataset metadata
        self.metadata = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load dataset metadata."""
        metadata_path = self.dataset_root / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "created": datetime.now().isoformat(),
                "datasets": {},
                "statistics": {}
            }
    
    def _save_metadata(self):
        """Save dataset metadata."""
        metadata_path = self.dataset_root / "dataset_metadata.json"
        self.metadata["updated"] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def create_dataset_from_database(
        self,
        dataset_name: str,
        min_samples_per_person: int = 5,
        max_samples_per_person: int = 50,
        augmentation_factor: int = 3
    ) -> Dict[str, Any]:
        """
        Create a training dataset from the database.
        
        Args:
            dataset_name: Name for the dataset
            min_samples_per_person: Minimum samples required per person
            max_samples_per_person: Maximum samples to use per person
            augmentation_factor: Number of augmented versions per image
            
        Returns:
            Dataset creation results
        """
        logger.info(f"Creating dataset '{dataset_name}' from database...")
        
        # Create dataset directory
        dataset_dir = self.dataset_root / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Get persons with sufficient samples
        persons = self.db_manager.list_persons()
        valid_persons = []
        
        for person in persons:
            embeddings = self.db_manager.get_person_embeddings(person.id)
            if len(embeddings) >= min_samples_per_person:
                valid_persons.append((person, embeddings))
        
        logger.info(f"Found {len(valid_persons)} persons with sufficient samples")
        
        dataset_stats = {
            "total_persons": len(valid_persons),
            "total_original_images": 0,
            "total_augmented_images": 0,
            "persons": {}
        }
        
        # Process each person
        for person, embeddings in valid_persons:
            person_dir = dataset_dir / person.name.replace(' ', '_')
            person_dir.mkdir(exist_ok=True)
            
            # Limit samples
            if len(embeddings) > max_samples_per_person:
                embeddings = np.random.choice(
                    embeddings, max_samples_per_person, replace=False
                )
            
            person_stats = {
                "original_images": 0,
                "augmented_images": 0,
                "total_quality_score": 0
            }
            
            image_count = 0
            for embedding in embeddings:
                # Load original image
                if embedding.source_image_path and os.path.exists(embedding.source_image_path):
                    try:
                        image = cv2.imread(embedding.source_image_path)
                        if image is None:
                            continue
                        
                        # Save original image
                        original_filename = f"{person.name}_{image_count:04d}_original.jpg"
                        original_path = person_dir / original_filename
                        cv2.imwrite(str(original_path), image)
                        
                        person_stats["original_images"] += 1
                        person_stats["total_quality_score"] += embedding.quality_score
                        
                        # Generate augmented versions
                        augmented_images = self.augmenter.augment_image(
                            image, augmentation_type="all"
                        )
                        
                        for aug_idx, aug_image in enumerate(augmented_images[1:]):  # Skip original
                            if aug_idx >= augmentation_factor:
                                break
                            
                            aug_filename = f"{person.name}_{image_count:04d}_aug_{aug_idx:02d}.jpg"
                            aug_path = person_dir / aug_filename
                            cv2.imwrite(str(aug_path), aug_image)
                            
                            person_stats["augmented_images"] += 1
                        
                        image_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing image for {person.name}: {str(e)}")
                        continue
            
            dataset_stats["persons"][person.name] = person_stats
            dataset_stats["total_original_images"] += person_stats["original_images"]
            dataset_stats["total_augmented_images"] += person_stats["augmented_images"]
        
        # Save dataset metadata
        dataset_info = {
            "name": dataset_name,
            "created": datetime.now().isoformat(),
            "statistics": dataset_stats,
            "config": {
                "min_samples_per_person": min_samples_per_person,
                "max_samples_per_person": max_samples_per_person,
                "augmentation_factor": augmentation_factor
            }
        }
        
        self.metadata["datasets"][dataset_name] = dataset_info
        self._save_metadata()
        
        # Save dataset-specific metadata
        dataset_metadata_path = dataset_dir / "metadata.json"
        with open(dataset_metadata_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Dataset '{dataset_name}' created successfully")
        logger.info(f"Total: {dataset_stats['total_original_images']} original + {dataset_stats['total_augmented_images']} augmented images")
        
        return dataset_info
    
    def validate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Validate dataset integrity and quality.
        
        Args:
            dataset_name: Name of dataset to validate
            
        Returns:
            Validation results
        """
        logger.info(f"Validating dataset '{dataset_name}'...")
        
        dataset_dir = self.dataset_root / dataset_name
        if not dataset_dir.exists():
            return {"status": "error", "message": "Dataset not found"}
        
        validation_results = {
            "status": "valid",
            "issues": [],
            "statistics": {
                "total_images": 0,
                "valid_images": 0,
                "corrupted_images": 0,
                "images_without_faces": 0,
                "persons": 0
            }
        }
        
        # Check each person directory
        for person_dir in dataset_dir.iterdir():
            if not person_dir.is_dir():
                continue
            
            validation_results["statistics"]["persons"] += 1
            person_images = 0
            
            for image_path in person_dir.glob("*.jpg"):
                validation_results["statistics"]["total_images"] += 1
                person_images += 1
                
                try:
                    # Try to load image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        validation_results["issues"].append(f"Corrupted image: {image_path}")
                        validation_results["statistics"]["corrupted_images"] += 1
                        continue
                    
                    # Check for faces
                    faces = self.face_detector.detect_faces(image)
                    if len(faces) == 0:
                        validation_results["issues"].append(f"No faces detected: {image_path}")
                        validation_results["statistics"]["images_without_faces"] += 1
                    else:
                        validation_results["statistics"]["valid_images"] += 1
                
                except Exception as e:
                    validation_results["issues"].append(f"Error processing {image_path}: {str(e)}")
                    validation_results["statistics"]["corrupted_images"] += 1
            
            # Check minimum images per person
            if person_images < 3:
                validation_results["issues"].append(f"Person {person_dir.name} has only {person_images} images")
        
        # Determine overall status
        if len(validation_results["issues"]) > 0:
            validation_results["status"] = "issues_found"
        
        logger.info(f"Dataset validation completed. Status: {validation_results['status']}")
        return validation_results
    
    def split_dataset(
        self,
        dataset_name: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """
        Split dataset into training, validation, and test sets.
        
        Args:
            dataset_name: Name of dataset to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Split information
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Split ratios must sum to 1.0")
        
        logger.info(f"Splitting dataset '{dataset_name}'...")
        
        dataset_dir = self.dataset_root / dataset_name
        splits_dir = dataset_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        # Create split directories
        for split in ["train", "val", "test"]:
            (splits_dir / split).mkdir(exist_ok=True)
        
        split_info = {
            "train": {"persons": 0, "images": 0},
            "val": {"persons": 0, "images": 0},
            "test": {"persons": 0, "images": 0}
        }
        
        # Process each person
        for person_dir in dataset_dir.iterdir():
            if not person_dir.is_dir() or person_dir.name == "splits":
                continue
            
            # Get all images for this person
            images = list(person_dir.glob("*.jpg"))
            if len(images) == 0:
                continue
            
            # Shuffle and split
            random.shuffle(images)
            
            n_train = int(len(images) * train_ratio)
            n_val = int(len(images) * val_ratio)
            
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Copy images to split directories
            for split, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
                if len(split_images) > 0:
                    split_person_dir = splits_dir / split / person_dir.name
                    split_person_dir.mkdir(exist_ok=True)
                    
                    for image_path in split_images:
                        dest_path = split_person_dir / image_path.name
                        shutil.copy2(image_path, dest_path)
                    
                    split_info[split]["images"] += len(split_images)
                    if len(split_images) > 0:
                        split_info[split]["persons"] += 1
        
        # Save split information
        split_metadata_path = splits_dir / "split_info.json"
        with open(split_metadata_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Dataset split completed: {split_info}")
        return split_info
    
    @performance_monitor.time_function("dataset_creation")
    def create_comprehensive_dataset(
        self,
        dataset_name: str,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive dataset with augmentation and validation.
        
        Args:
            dataset_name: Name for the dataset
            config: Configuration parameters
            
        Returns:
            Complete dataset creation results
        """
        default_config = {
            "min_samples_per_person": 5,
            "max_samples_per_person": 50,
            "augmentation_factor": 3,
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "test_ratio": 0.1
        }
        
        if config:
            default_config.update(config)
        
        results = {}
        
        # Create dataset
        results["creation"] = self.create_dataset_from_database(
            dataset_name,
            default_config["min_samples_per_person"],
            default_config["max_samples_per_person"],
            default_config["augmentation_factor"]
        )
        
        # Validate dataset
        results["validation"] = self.validate_dataset(dataset_name)
        
        # Split dataset
        results["splits"] = self.split_dataset(
            dataset_name,
            default_config["train_ratio"],
            default_config["val_ratio"],
            default_config["test_ratio"]
        )
        
        return results

def create_training_dataset(
    dataset_name: str,
    dataset_root: str = "datasets",
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive training dataset.
    
    Args:
        dataset_name: Name for the dataset
        dataset_root: Root directory for datasets
        config: Configuration parameters
        
    Returns:
        Dataset creation results
    """
    manager = DatasetManager(dataset_root)
    results = manager.create_comprehensive_dataset(dataset_name, config)
    
    return results

if __name__ == "__main__":
    # Example usage
    config = {
        "min_samples_per_person": 3,
        "max_samples_per_person": 20,
        "augmentation_factor": 2
    }
    
    results = create_training_dataset("face_recognition_v1", config=config)
    print(f"Dataset creation completed: {results['creation']['statistics']}")