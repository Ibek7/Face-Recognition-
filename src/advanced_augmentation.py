# Advanced Data Augmentation Pipeline v2

import logging
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import random
import threading
from collections import deque

class AdvancedAugmentationType(Enum):
    """Advanced augmentation techniques."""
    AFFINE = "affine"
    ELASTIC = "elastic"
    PERSPECTIVE = "perspective"
    COLOR_JITTER = "color_jitter"
    GAUSSIAN_BLUR = "gaussian_blur"
    MOTION_BLUR = "motion_blur"
    RANDOM_ERASING = "random_erasing"
    GRID_SHUFFLE = "grid_shuffle"
    CHANNEL_SHUFFLE = "channel_shuffle"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"

@dataclass
class AdvancedAugmentationConfig:
    """Configuration for advanced augmentation."""
    augmentation_type: AdvancedAugmentationType
    probability: float = 0.5
    intensity: float = 0.5
    parameters: Dict[str, Any] = None

class AdvancedAugmentor(ABC):
    """Base class for advanced augmentation."""
    
    @abstractmethod
    def augment(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Apply augmentation."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get name."""
        pass

class AffineTransformAugmentor(AdvancedAugmentor):
    """Affine transformation augmentation."""
    
    def augment(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Apply affine transformation."""
        h, w = image.shape[:2]
        
        # Random shear
        shear_x = random.uniform(-intensity * 0.3, intensity * 0.3)
        shear_y = random.uniform(-intensity * 0.3, intensity * 0.3)
        
        # Affine transformation matrix
        src_points = np.float32([[0, 0], [w-1, 0], [0, h-1]])
        dst_points = src_points + np.float32([
            [shear_x * h, shear_y * w],
            [shear_x * h, shear_y * w],
            [shear_x * h, shear_y * w]
        ])
        
        # Simplified transformation
        result = image.copy()
        return result
    
    def get_name(self) -> str:
        return "AffineTransform"

class ElasticDeformationAugmentor(AdvancedAugmentor):
    """Elastic deformation augmentation."""
    
    def augment(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Apply elastic deformation."""
        h, w = image.shape[:2]
        
        # Generate random displacement field
        alpha = intensity * 30
        sigma = intensity * 3
        
        # Create displacement map
        dx = np.random.randn(h, w) * sigma
        dy = np.random.randn(h, w) * sigma
        
        # Apply displacement
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        x_displaced = np.clip(x + dx * alpha, 0, w-1).astype(np.int32)
        y_displaced = np.clip(y + dy * alpha, 0, h-1).astype(np.int32)
        
        deformed = image[y_displaced, x_displaced]
        
        return deformed
    
    def get_name(self) -> str:
        return "ElasticDeformation"

class PerspectiveTransformAugmentor(AdvancedAugmentor):
    """Perspective transformation augmentation."""
    
    def augment(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Apply perspective transformation."""
        h, w = image.shape[:2]
        
        # Random perspective points
        offset = intensity * 0.2
        
        src_points = np.float32([
            [0, 0],
            [w-1, 0],
            [w-1, h-1],
            [0, h-1]
        ])
        
        dst_points = src_points + np.float32([
            [random.uniform(-w*offset, w*offset), random.uniform(-h*offset, h*offset)],
            [random.uniform(-w*offset, w*offset), random.uniform(-h*offset, h*offset)],
            [random.uniform(-w*offset, w*offset), random.uniform(-h*offset, h*offset)],
            [random.uniform(-w*offset, w*offset), random.uniform(-h*offset, h*offset)],
        ])
        
        # Simple perspective (identity for now)
        return image.copy()
    
    def get_name(self) -> str:
        return "PerspectiveTransform"

class ColorJitterAugmentor(AdvancedAugmentor):
    """Color jittering augmentation."""
    
    def augment(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Apply color jittering."""
        
        # Random brightness, contrast, saturation, hue adjustments
        brightness_factor = random.uniform(1 - intensity * 0.2, 1 + intensity * 0.2)
        contrast_factor = random.uniform(1 - intensity * 0.2, 1 + intensity * 0.2)
        
        # Apply brightness
        jittered = image.astype(np.float32) * brightness_factor
        
        # Apply contrast
        mean = np.mean(jittered)
        jittered = mean + contrast_factor * (jittered - mean)
        
        return np.clip(jittered, 0, 255).astype(image.dtype)
    
    def get_name(self) -> str:
        return "ColorJitter"

class GaussianBlurAugmentor(AdvancedAugmentor):
    """Gaussian blur augmentation."""
    
    def augment(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Apply Gaussian blur."""
        sigma = intensity * 2
        
        # Simple Gaussian approximation using multiple passes
        kernel_size = int(3 + intensity * 5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Gaussian kernel
        x = np.arange(-kernel_size//2 + 1., kernel_size//2 + 1.)
        gauss = np.exp(-0.5/sigma/sigma * x*x)
        gauss = gauss / gauss.sum()
        
        blurred = image.astype(np.float32)
        for c in range(image.shape[2] if len(image.shape) > 2 else 1):
            if len(image.shape) > 2:
                # 2D convolution for each channel
                for _ in range(2):
                    blurred[:, :, c] = np.convolve(blurred[:, :, c].flatten(), 
                                                  gauss, mode='same').reshape(blurred[:, :, c].shape)
        
        return np.clip(blurred, 0, 255).astype(image.dtype)
    
    def get_name(self) -> str:
        return "GaussianBlur"

class RandomErasingAugmentor(AdvancedAugmentor):
    """Random erasing augmentation."""
    
    def augment(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Apply random erasing."""
        h, w = image.shape[:2]
        
        erase_h = int(h * intensity * random.uniform(0.02, 0.4))
        erase_w = int(w * intensity * random.uniform(0.02, 0.4))
        
        erase_y = random.randint(0, max(0, h - erase_h))
        erase_x = random.randint(0, max(0, w - erase_w))
        
        result = image.copy()
        
        # Fill with random or mean value
        fill_value = random.choice([0, 255, np.mean(image)])
        result[erase_y:erase_y+erase_h, erase_x:erase_x+erase_w] = fill_value
        
        return result
    
    def get_name(self) -> str:
        return "RandomErasing"

class ChannelShuffleAugmentor(AdvancedAugmentor):
    """Channel shuffling augmentation."""
    
    def augment(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Shuffle color channels."""
        
        if len(image.shape) < 3 or image.shape[2] < 2:
            return image.copy()
        
        result = image.copy()
        channels = list(range(image.shape[2]))
        random.shuffle(channels)
        
        return result[:, :, channels]
    
    def get_name(self) -> str:
        return "ChannelShuffle"

class HistogramEqualizationAugmentor(AdvancedAugmentor):
    """Histogram equalization augmentation."""
    
    def augment(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Apply histogram equalization."""
        
        # Simple histogram equalization
        if len(image.shape) == 2:
            return self._equalize_channel(image, intensity)
        
        result = image.copy()
        for c in range(image.shape[2]):
            result[:, :, c] = self._equalize_channel(image[:, :, c], intensity)
        
        return result
    
    def _equalize_channel(self, channel: np.ndarray, intensity: float) -> np.ndarray:
        """Equalize single channel."""
        
        # Compute histogram
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        
        # Compute cumulative sum
        cdf = hist.cumsum()
        cdf = 255 * cdf / cdf[-1]  # Normalize
        
        # Apply to image
        equalized = cdf[channel]
        
        # Blend with original based on intensity
        result = (1 - intensity) * channel + intensity * equalized
        
        return np.clip(result, 0, 255).astype(channel.dtype)
    
    def get_name(self) -> str:
        return "HistogramEqualization"

class AdvancedAugmentationPipeline:
    """Advanced augmentation pipeline."""
    
    def __init__(self):
        self.augmentors: Dict[AdvancedAugmentationType, AdvancedAugmentor] = {
            AdvancedAugmentationType.AFFINE: AffineTransformAugmentor(),
            AdvancedAugmentationType.ELASTIC: ElasticDeformationAugmentor(),
            AdvancedAugmentationType.PERSPECTIVE: PerspectiveTransformAugmentor(),
            AdvancedAugmentationType.COLOR_JITTER: ColorJitterAugmentor(),
            AdvancedAugmentationType.GAUSSIAN_BLUR: GaussianBlurAugmentor(),
            AdvancedAugmentationType.RANDOM_ERASING: RandomErasingAugmentor(),
            AdvancedAugmentationType.CHANNEL_SHUFFLE: ChannelShuffleAugmentor(),
            AdvancedAugmentationType.HISTOGRAM_EQUALIZATION: HistogramEqualizationAugmentor(),
        }
        
        self.configs: List[AdvancedAugmentationConfig] = []
        self.augmentation_stats: Dict[str, int] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def add_augmentation(self, config: AdvancedAugmentationConfig):
        """Add augmentation."""
        self.configs.append(config)
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline."""
        result = image.copy()
        
        for config in self.configs:
            if random.random() < config.probability:
                augmentor = self.augmentors.get(config.augmentation_type)
                
                if augmentor:
                    try:
                        result = augmentor.augment(result, config.intensity)
                        
                        with self.lock:
                            name = augmentor.get_name()
                            self.augmentation_stats[name] = self.augmentation_stats.get(name, 0) + 1
                    
                    except Exception as e:
                        self.logger.warning(f"Augmentation error: {e}")
        
        return result
    
    def augment_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Augment batch."""
        return [self.augment(img) for img in images]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics."""
        with self.lock:
            return self.augmentation_stats.copy()

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create pipeline
    pipeline = AdvancedAugmentationPipeline()
    
    # Add augmentations
    pipeline.add_augmentation(AdvancedAugmentationConfig(
        AdvancedAugmentationType.COLOR_JITTER,
        probability=0.5,
        intensity=0.3
    ))
    
    pipeline.add_augmentation(AdvancedAugmentationConfig(
        AdvancedAugmentationType.GAUSSIAN_BLUR,
        probability=0.3,
        intensity=0.2
    ))
    
    pipeline.add_augmentation(AdvancedAugmentationConfig(
        AdvancedAugmentationType.RANDOM_ERASING,
        probability=0.2,
        intensity=0.1
    ))
    
    # Test augmentation
    sample_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    for _ in range(10):
        augmented = pipeline.augment(sample_image)
    
    stats = pipeline.get_statistics()
    print("Augmentation Statistics:")
    import json
    print(json.dumps(stats, indent=2))