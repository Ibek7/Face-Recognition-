#!/usr/bin/env python3
"""
CLI tool for batch face detection and preprocessing.
Processes directories of images and extracts/analyzes faces.
"""

import argparse
import cv2
import os
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import FaceProcessingPipeline

def main():
    parser = argparse.ArgumentParser(description='Batch Face Detection and Preprocessing')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing images to process')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save processed faces')
    parser.add_argument('--quality-threshold', type=float, default=100.0,
                        help='Minimum quality score for faces')
    parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224],
                        help='Target size for normalized faces (width height)')
    parser.add_argument('--formats', nargs='+', default=['jpg', 'jpeg', 'png'],
                        help='Image formats to process')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FaceProcessingPipeline(
        target_size=tuple(args.target_size),
        quality_threshold=args.quality_threshold
    )
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process images
    input_path = Path(args.input_dir)
    total_stats = {'images': 0, 'faces': 0, 'quality_faces': 0}
    
    for ext in args.formats:
        for img_file in input_path.glob(f'*.{ext}'):
            print(f"Processing: {img_file}")
            
            try:
                results = pipeline.process_image(str(img_file))
                total_stats['images'] += 1
                total_stats['faces'] += results['metadata']['total_faces_detected']
                total_stats['quality_faces'] += results['metadata']['quality_faces']
                
                # Save processed faces
                for face_data in results['faces']:
                    face_filename = f"{img_file.stem}_face_{face_data['face_id']}.jpg"
                    face_path = output_path / face_filename
                    
                    # Convert normalized face back to uint8 for saving
                    face_img = (face_data['normalized_face'] * 255).astype('uint8')
                    cv2.imwrite(str(face_path), face_img)
                
                # Save metadata
                metadata_file = output_path / f"{img_file.stem}_metadata.json"
                with open(metadata_file, 'w') as f:
                    # Remove numpy arrays for JSON serialization
                    clean_results = {
                        'source': results['source'],
                        'input_shape': results['input_shape'],
                        'metadata': results['metadata'],
                        'face_count': len(results['faces'])
                    }
                    json.dump(clean_results, f, indent=2)
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    print(f"\nProcessing complete!")
    print(f"Images processed: {total_stats['images']}")
    print(f"Total faces detected: {total_stats['faces']}")
    print(f"Quality faces saved: {total_stats['quality_faces']}")

if __name__ == '__main__':
    main()