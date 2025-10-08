#!/usr/bin/env python3
"""
Generate face embeddings from a dataset and save for later use.
Creates a searchable database of face embeddings with metadata.
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import sys
import os
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from embeddings import FaceEmbeddingManager

class EmbeddingDatabase:
    """Database for storing and managing face embeddings."""
    
    def __init__(self, db_path: str = "face_embeddings.db"):
        """Initialize embedding database."""
        self.db_path = Path(db_path)
        self.embeddings = []
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'total_embeddings': 0,
            'encoder_type': None,
            'source_images': []
        }
        
    def add_embedding(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add an embedding with metadata to the database."""
        embedding_entry = {
            'id': len(self.embeddings),
            'embedding': embedding,
            'metadata': metadata,
            'added_at': datetime.now().isoformat()
        }
        self.embeddings.append(embedding_entry)
        self.metadata['total_embeddings'] = len(self.embeddings)
    
    def save(self) -> None:
        """Save database to disk."""
        db_data = {
            'metadata': self.metadata,
            'embeddings': self.embeddings
        }
        
        # Save as pickle for numpy arrays
        with open(self.db_path, 'wb') as f:
            pickle.dump(db_data, f)
            
        # Also save metadata as JSON for readability
        metadata_path = self.db_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load(self) -> bool:
        """Load database from disk."""
        try:
            with open(self.db_path, 'rb') as f:
                db_data = pickle.load(f)
                self.metadata = db_data['metadata']
                self.embeddings = db_data['embeddings']
            return True
        except FileNotFoundError:
            return False
    
    def search_similar(self, 
                      query_embedding: np.ndarray, 
                      top_k: int = 5,
                      threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for similar embeddings in the database."""
        from similarity import FaceSimilarity, DistanceMetric
        
        similarity_calc = FaceSimilarity(DistanceMetric.EUCLIDEAN)
        results = []
        
        for entry in self.embeddings:
            similarity = similarity_calc.calculate_similarity(query_embedding, entry['embedding'])
            
            if similarity >= threshold:
                result = {
                    'id': entry['id'],
                    'similarity': similarity,
                    'metadata': entry['metadata'],
                    'added_at': entry['added_at']
                }
                results.append(result)
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

def process_dataset(input_dir: str, 
                   output_db: str,
                   encoder_type: str = "simple",
                   batch_size: int = 10) -> None:
    """
    Process a dataset of images and create embedding database.
    
    Args:
        input_dir: Directory containing images
        output_db: Path for output database
        encoder_type: Type of encoder to use
        batch_size: Number of images to process at once
    """
    input_path = Path(input_dir)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(input_path.glob(ext))
        all_images.extend(input_path.rglob(ext))  # Recursive search
    
    if not all_images:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(all_images)} images to process")
    
    # Initialize components
    embedding_manager = FaceEmbeddingManager(encoder_type=encoder_type)
    database = EmbeddingDatabase(output_db)
    
    # Set encoder type in metadata
    database.metadata['encoder_type'] = encoder_type
    
    # Process images with progress bar
    with tqdm(total=len(all_images), desc="Processing images") as pbar:
        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i + batch_size]
            
            for image_path in batch:
                try:
                    # Generate embeddings
                    embedding_data = embedding_manager.generate_embeddings_from_image(str(image_path))
                    
                    # Add to database
                    for emb_info in embedding_data['embeddings']:
                        metadata = {
                            'source_image': str(image_path),
                            'face_id': emb_info['face_id'],
                            'quality_score': emb_info['quality_score'],
                            'embedding_dim': emb_info['embedding_dim'],
                            'relative_path': str(image_path.relative_to(input_path))
                        }
                        
                        database.add_embedding(emb_info['embedding'], metadata)
                    
                    # Track source images
                    database.metadata['source_images'].append(str(image_path))
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                
                pbar.update(1)
    
    # Save database
    database.save()
    print(f"\nDatabase saved to: {output_db}")
    print(f"Total embeddings: {database.metadata['total_embeddings']}")
    print(f"Metadata file: {Path(output_db).with_suffix('.json')}")

def search_database(db_path: str, query_image: str, top_k: int = 5) -> None:
    """
    Search the embedding database for similar faces.
    
    Args:
        db_path: Path to embedding database
        query_image: Path to query image
        top_k: Number of results to return
    """
    # Load database
    database = EmbeddingDatabase(db_path)
    if not database.load():
        print(f"Could not load database: {db_path}")
        return
    
    print(f"Loaded database with {database.metadata['total_embeddings']} embeddings")
    
    # Generate embedding for query image
    encoder_type = database.metadata.get('encoder_type', 'simple')
    embedding_manager = FaceEmbeddingManager(encoder_type=encoder_type)
    
    try:
        query_data = embedding_manager.generate_embeddings_from_image(query_image)
        
        if not query_data['embeddings']:
            print(f"No faces found in query image: {query_image}")
            return
        
        # Search for each face in query image
        for i, emb_info in enumerate(query_data['embeddings']):
            print(f"\nSearching for face {i} (quality: {emb_info['quality_score']:.2f}):")
            
            results = database.search_similar(
                emb_info['embedding'], 
                top_k=top_k, 
                threshold=0.3
            )
            
            if results:
                print(f"Found {len(results)} similar faces:")
                for j, result in enumerate(results):
                    print(f"  {j+1}. {result['metadata']['source_image']} "
                          f"(similarity: {result['similarity']:.3f}, "
                          f"quality: {result['metadata']['quality_score']:.2f})")
            else:
                print("  No similar faces found")
    
    except Exception as e:
        print(f"Error processing query image: {e}")

def main():
    parser = argparse.ArgumentParser(description='Face Embedding Database Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create database command
    create_parser = subparsers.add_parser('create', help='Create embedding database from dataset')
    create_parser.add_argument('--input-dir', type=str, required=True,
                              help='Directory containing images')
    create_parser.add_argument('--output-db', type=str, required=True,
                              help='Output database file path')
    create_parser.add_argument('--encoder', type=str, default='simple',
                              choices=['simple', 'dlib'],
                              help='Encoder type to use')
    create_parser.add_argument('--batch-size', type=int, default=10,
                              help='Batch size for processing')
    
    # Search database command
    search_parser = subparsers.add_parser('search', help='Search database for similar faces')
    search_parser.add_argument('--database', type=str, required=True,
                              help='Path to embedding database')
    search_parser.add_argument('--query-image', type=str, required=True,
                              help='Query image path')
    search_parser.add_argument('--top-k', type=int, default=5,
                              help='Number of results to return')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        process_dataset(
            args.input_dir,
            args.output_db,
            args.encoder,
            args.batch_size
        )
    elif args.command == 'search':
        search_database(
            args.database,
            args.query_image,
            args.top_k
        )
    else:
        parser.print_help()

if __name__ == '__main__':
    main()