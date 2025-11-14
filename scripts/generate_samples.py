#!/usr/bin/env python3
"""
Generate sample test images with faces for testing and development.
Uses simple geometric shapes to create synthetic face-like images.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse


def create_face_image(size=(200, 200), name="Sample", seed=None):
    """
    Create a simple synthetic face image.
    
    Args:
        size: Image size (width, height)
        name: Name to display on image
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create image with random background color
    bg_color = tuple(np.random.randint(200, 256, 3))
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw face (circle)
    face_color = tuple(np.random.randint(180, 230, 3))
    center_x, center_y = size[0] // 2, size[1] // 2
    radius = min(size) // 3
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        fill=face_color,
        outline=(100, 100, 100),
        width=2
    )
    
    # Draw eyes
    eye_color = (50, 50, 50)
    eye_radius = radius // 8
    left_eye_x = center_x - radius // 2
    right_eye_x = center_x + radius // 2
    eye_y = center_y - radius // 4
    
    draw.ellipse(
        [left_eye_x - eye_radius, eye_y - eye_radius, 
         left_eye_x + eye_radius, eye_y + eye_radius],
        fill=eye_color
    )
    draw.ellipse(
        [right_eye_x - eye_radius, eye_y - eye_radius, 
         right_eye_x + eye_radius, eye_y + eye_radius],
        fill=eye_color
    )
    
    # Draw mouth (arc)
    mouth_y = center_y + radius // 3
    draw.arc(
        [center_x - radius // 2, mouth_y - radius // 4, 
         center_x + radius // 2, mouth_y + radius // 4],
        start=0,
        end=180,
        fill=(100, 50, 50),
        width=3
    )
    
    # Add name label
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    text_bbox = draw.textbbox((0, 0), name, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (size[0] - text_width) // 2
    text_y = size[1] - 30
    
    draw.text((text_x, text_y), name, fill=(0, 0, 0), font=font)
    
    return img


def generate_sample_dataset(output_dir: Path, num_persons: int = 5, images_per_person: int = 3):
    """Generate a complete sample dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_names = [
        "Alice", "Bob", "Charlie", "Diana", "Eve",
        "Frank", "Grace", "Henry", "Iris", "Jack"
    ]
    
    print(f"Generating sample dataset in: {output_dir}")
    print(f"  Persons: {num_persons}")
    print(f"  Images per person: {images_per_person}")
    print()
    
    for i in range(num_persons):
        name = sample_names[i % len(sample_names)]
        person_dir = output_dir / name.lower()
        person_dir.mkdir(exist_ok=True)
        
        for j in range(images_per_person):
            # Create image with slight variations
            seed = i * 1000 + j
            img = create_face_image(
                size=(200, 200),
                name=name,
                seed=seed
            )
            
            # Save image
            img_path = person_dir / f"{name.lower()}_{j+1}.jpg"
            img.save(img_path, "JPEG", quality=95)
            
        print(f"✓ Created {images_per_person} images for {name} in {person_dir}")
    
    print(f"\n✓ Sample dataset generated successfully!")
    print(f"  Total persons: {num_persons}")
    print(f"  Total images: {num_persons * images_per_person}")
    print(f"  Location: {output_dir.absolute()}")


def generate_test_images(output_dir: Path, count: int = 10):
    """Generate standalone test images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {count} test images in: {output_dir}")
    
    for i in range(count):
        img = create_face_image(
            size=(200, 200),
            name=f"Test{i+1}",
            seed=i * 100
        )
        
        img_path = output_dir / f"test_{i+1:03d}.jpg"
        img.save(img_path, "JPEG", quality=95)
        
        if (i + 1) % 5 == 0:
            print(f"  Created {i + 1}/{count} images...")
    
    print(f"✓ Generated {count} test images")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate sample images for face recognition testing"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_images",
        help="Output directory for generated images"
    )
    
    parser.add_argument(
        "--persons",
        type=int,
        default=5,
        help="Number of persons to generate"
    )
    
    parser.add_argument(
        "--images-per-person",
        type=int,
        default=3,
        help="Number of images per person"
    )
    
    parser.add_argument(
        "--test-images",
        type=int,
        default=0,
        help="Number of standalone test images to generate"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    print("=" * 60)
    print("Sample Image Generator")
    print("=" * 60)
    print()
    
    # Generate dataset
    if args.persons > 0:
        generate_sample_dataset(
            output_path,
            num_persons=args.persons,
            images_per_person=args.images_per_person
        )
        print()
    
    # Generate test images
    if args.test_images > 0:
        test_dir = output_path.parent / "test_images"
        generate_test_images(test_dir, count=args.test_images)
        print()
    
    print("Done!")


if __name__ == "__main__":
    main()
