#!/usr/bin/env python3
"""
Database seeding script for Face Recognition System.
Populates the database with sample persons and embeddings for testing.
"""

import sys
import os
from pathlib import Path
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import DatabaseManager
from src.embeddings import FaceEmbeddingManager


def create_sample_persons(db: DatabaseManager, count: int = 10):
    """Create sample person records."""
    print(f"Creating {count} sample persons...")
    
    sample_names = [
        ("John Doe", "Sample employee"),
        ("Jane Smith", "Engineering team"),
        ("Alice Johnson", "Marketing department"),
        ("Bob Williams", "Sales representative"),
        ("Charlie Brown", "IT support"),
        ("Diana Prince", "HR manager"),
        ("Eve Anderson", "Finance analyst"),
        ("Frank Miller", "Product manager"),
        ("Grace Lee", "Designer"),
        ("Henry Wilson", "Developer"),
    ]
    
    persons = []
    for i in range(min(count, len(sample_names))):
        name, description = sample_names[i]
        person = db.add_person(name=name, description=description)
        persons.append(person)
        print(f"  ✓ Created: {name} (ID: {person.id})")
    
    return persons


def create_sample_embeddings(db: DatabaseManager, persons: list, embeddings_per_person: int = 3):
    """Create sample face embeddings for each person."""
    print(f"\nCreating {embeddings_per_person} embeddings per person...")
    
    encoder_types = ["simple", "dlib", "facenet"]
    
    for person in persons:
        for i in range(embeddings_per_person):
            # Generate random embedding (512-dimensional)
            embedding = np.random.randn(512).astype(np.float32)
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            # Add to database
            face_embedding = db.add_face_embedding(
                embedding=embedding,
                person_id=person.id,
                source_image_path=f"sample_images/{person.name.lower().replace(' ', '_')}_{i}.jpg",
                quality_score=np.random.uniform(0.7, 0.99),
                encoder_type=encoder_types[i % len(encoder_types)],
                face_confidence=np.random.uniform(0.85, 0.99)
            )
            
        print(f"  ✓ Added {embeddings_per_person} embeddings for {person.name}")


def create_sample_recognition_results(db: DatabaseManager, persons: list, count: int = 20):
    """Create sample recognition history."""
    print(f"\nCreating {count} sample recognition results...")
    
    for i in range(count):
        person = np.random.choice(persons) if np.random.random() > 0.2 else None
        
        result = db.add_recognition_result(
            person_id=person.id if person else None,
            confidence_score=np.random.uniform(0.5, 0.99) if person else np.random.uniform(0.1, 0.4),
            source_image_path=f"test_images/test_{i}.jpg",
            processing_time=np.random.uniform(0.05, 0.5),
            distance_score=np.random.uniform(0.1, 0.8),
            similarity_metric="euclidean",
            source_type="image"
        )
    
    print(f"  ✓ Created {count} recognition results")


def print_statistics(db: DatabaseManager):
    """Print database statistics."""
    stats = db.get_recognition_stats()
    
    print("\n" + "=" * 60)
    print("Database Statistics")
    print("=" * 60)
    print(f"Total Persons: {stats['total_persons']}")
    print(f"Total Embeddings: {stats['total_embeddings']}")
    print(f"Total Recognitions: {stats['total_recognitions']}")
    print(f"Successful Recognitions: {stats['successful_recognitions']}")
    print(f"Average Processing Time: {stats['avg_processing_time']:.3f}s")
    print("=" * 60 + "\n")


def main():
    """Main seeding function."""
    print("=" * 60)
    print("Face Recognition Database Seeding")
    print("=" * 60 + "\n")
    
    # Get database URL from environment or use default
    db_url = os.getenv("DATABASE_URL", "sqlite:///face_recognition.db")
    print(f"Database: {db_url}\n")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Seed database with sample data")
    parser.add_argument("--persons", type=int, default=10, help="Number of persons to create")
    parser.add_argument("--embeddings", type=int, default=3, help="Embeddings per person")
    parser.add_argument("--results", type=int, default=20, help="Recognition results to create")
    parser.add_argument("--clear", action="store_true", help="Clear database before seeding")
    args = parser.parse_args()
    
    # Initialize database
    db = DatabaseManager(db_url)
    
    # Clear database if requested
    if args.clear:
        print("WARNING: Clearing database...")
        response = input("Are you sure? (yes/no): ")
        if response.lower() == "yes":
            # This would require implementing a clear method
            print("Database cleared (not implemented - manually delete the database file)")
        else:
            print("Skipping clear operation")
        print()
    
    # Create sample data
    persons = create_sample_persons(db, args.persons)
    create_sample_embeddings(db, persons, args.embeddings)
    create_sample_recognition_results(db, persons, args.results)
    
    # Print statistics
    print_statistics(db)
    
    print("✓ Database seeding complete!\n")


if __name__ == "__main__":
    main()
