#!/usr/bin/env python3
"""
Comprehensive CLI for face recognition system.
Provides unified interface for all face recognition operations.
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database import DatabaseManager
from realtime import RealTimeFaceRecognizer
from embeddings import FaceEmbeddingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def setup_database(args):
    """Initialize database and show setup information."""
    db_manager = DatabaseManager(f"sqlite:///{args.database}")
    
    print("Face Recognition Database Setup")
    print("=" * 40)
    
    # Show statistics
    stats = db_manager.get_recognition_stats()
    print(f"Persons: {stats['total_persons']}")
    print(f"Embeddings: {stats['total_embeddings']}")
    print(f"Recognitions: {stats['total_recognitions']}")
    
    if stats['embeddings_by_encoder']:
        print("\\nEmbeddings by encoder:")
        for encoder, count in stats['embeddings_by_encoder'].items():
            print(f"  {encoder}: {count}")
    
    print(f"\\nDatabase location: {args.database}")
    print("Database is ready for use!")

def add_person(args):
    """Add a new person to the database."""
    db_manager = DatabaseManager(f"sqlite:///{args.database}")
    
    # Check if person already exists
    existing = db_manager.get_person_by_name(args.name)
    if existing:
        print(f"Person '{args.name}' already exists (ID: {existing.id})")
        return
    
    # Add new person
    person = db_manager.add_person(args.name, args.description or "")
    print(f"Added person: {person.name} (ID: {person.id})")

def add_face_images(args):
    """Add face images for a person."""
    db_manager = DatabaseManager(f"sqlite:///{args.database}")
    embedding_manager = FaceEmbeddingManager(encoder_type=args.encoder)
    
    # Get person
    person = db_manager.get_person_by_name(args.person_name)
    if not person:
        print(f"Person '{args.person_name}' not found. Add them first.")
        return
    
    # Process images
    image_path = Path(args.image_path)
    
    if image_path.is_file():
        # Single image
        images = [image_path]
    elif image_path.is_dir():
        # Directory of images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(image_path.glob(ext))
    else:
        print(f"Invalid path: {args.image_path}")
        return
    
    print(f"Processing {len(images)} images for {person.name}...")
    
    total_embeddings = 0
    for img_path in images:
        try:
            embedding_data = embedding_manager.generate_embeddings_from_image(str(img_path))
            
            for emb_info in embedding_data['embeddings']:
                db_manager.add_face_embedding(
                    embedding=emb_info['embedding'],
                    person_id=person.id,
                    source_image_path=str(img_path),
                    quality_score=emb_info['quality_score'],
                    encoder_type=args.encoder,
                    source_type="image"
                )
                total_embeddings += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Added {total_embeddings} face embeddings for {person.name}")

def recognize_image(args):
    """Recognize faces in an image."""
    db_manager = DatabaseManager(f"sqlite:///{args.database}")
    embedding_manager = FaceEmbeddingManager(encoder_type=args.encoder)
    
    try:
        # Generate embeddings for query image
        embedding_data = embedding_manager.generate_embeddings_from_image(args.image_path)
        
        if not embedding_data['embeddings']:
            print(f"No faces found in {args.image_path}")
            return
        
        print(f"Found {len(embedding_data['embeddings'])} faces in image:")
        print()
        
        # Search for each face
        for i, emb_info in enumerate(embedding_data['embeddings']):
            print(f"Face {i + 1}:")
            print(f"  Quality score: {emb_info['quality_score']:.2f}")
            
            # Search database
            similar_faces = db_manager.search_similar_faces(
                emb_info['embedding'],
                threshold=args.threshold,
                top_k=args.top_k
            )
            
            if similar_faces:
                print(f"  Top matches:")
                for j, (face_embedding, similarity) in enumerate(similar_faces):
                    person = db_manager.get_person(face_embedding.person_id)
                    person_name = person.name if person else "Unknown"
                    print(f"    {j+1}. {person_name} (similarity: {similarity:.3f})")
            else:
                print(f"  No matches found (threshold: {args.threshold})")
            print()
    
    except Exception as e:
        print(f"Error processing image: {e}")

def start_webcam(args):
    """Start real-time webcam recognition."""
    recognizer = RealTimeFaceRecognizer(
        db_path=args.database,
        encoder_type=args.encoder,
        confidence_threshold=args.threshold
    )
    
    try:
        recognizer.start_webcam(camera_id=args.camera, display=True)
    except KeyboardInterrupt:
        print("\\nStopping webcam recognition...")
    except Exception as e:
        print(f"Error starting webcam: {e}")

def add_person_webcam(args):
    """Add a person using webcam capture."""
    recognizer = RealTimeFaceRecognizer(
        db_path=args.database,
        encoder_type=args.encoder
    )
    
    success = recognizer.add_person_from_webcam(
        person_name=args.name,
        camera_id=args.camera,
        num_samples=args.samples
    )
    
    if success:
        print(f"Successfully added person: {args.name}")
    else:
        print(f"Failed to add person: {args.name}")

def list_persons(args):
    """List all persons in the database."""
    db_manager = DatabaseManager(f"sqlite:///{args.database}")
    
    persons = db_manager.list_persons()
    
    if not persons:
        print("No persons found in database.")
        return
    
    print(f"Found {len(persons)} persons:")
    print()
    
    for person in persons:
        embeddings = db_manager.get_face_embeddings(person_id=person.id)
        print(f"ID: {person.id}")
        print(f"Name: {person.name}")
        print(f"Description: {person.description or 'None'}")
        print(f"Face embeddings: {len(embeddings)}")
        print(f"Added: {person.created_at}")
        print("-" * 40)

def run_benchmark(args):
    """Run performance benchmark."""
    from benchmark import FaceRecognitionBenchmark
    
    if not Path(args.test_images).exists():
        print(f"Test images directory not found: {args.test_images}")
        return
    
    benchmark = FaceRecognitionBenchmark(args.output_dir)
    
    results = {}
    
    # Benchmark encoders
    print("Running encoder benchmark...")
    encoder_results = benchmark.benchmark_encoders(args.test_images)
    if encoder_results:
        results['encoders'] = encoder_results
    
    # Generate report
    if results:
        benchmark.generate_report(results)
        print(f"Benchmark results saved to: {args.output_dir}")
    else:
        print("No benchmark results generated.")

def cleanup_database(args):
    """Clean up old recognition results."""
    db_manager = DatabaseManager(f"sqlite:///{args.database}")
    
    if args.days:
        deleted_count = db_manager.cleanup_old_results(args.days)
        print(f"Deleted {deleted_count} old recognition results (older than {args.days} days)")
    
    # Show updated stats
    stats = db_manager.get_recognition_stats()
    print(f"Current recognitions in database: {stats['total_recognitions']}")

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Face Recognition System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s setup --database faces.db
  %(prog)s add-person --name "John Doe" --description "Employee"
  %(prog)s add-faces --person-name "John Doe" --image-path ./john_photos/
  %(prog)s recognize --image-path photo.jpg --threshold 0.7
  %(prog)s webcam --camera 0 --threshold 0.6
  %(prog)s add-person-webcam --name "Jane Doe" --samples 5
  %(prog)s benchmark --test-images ./test_data/
        """
    )
    
    # Global arguments
    parser.add_argument('--database', type=str, default='face_recognition.db',
                       help='Database file path')
    parser.add_argument('--encoder', type=str, default='simple',
                       choices=['simple', 'dlib'],
                       help='Face encoder type')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup database')
    setup_parser.set_defaults(func=setup_database)
    
    # Add person command
    add_person_parser = subparsers.add_parser('add-person', help='Add a new person')
    add_person_parser.add_argument('--name', type=str, required=True,
                                  help='Person name')
    add_person_parser.add_argument('--description', type=str,
                                  help='Person description')
    add_person_parser.set_defaults(func=add_person)
    
    # Add face images command
    add_faces_parser = subparsers.add_parser('add-faces', help='Add face images for a person')
    add_faces_parser.add_argument('--person-name', type=str, required=True,
                                 help='Person name')
    add_faces_parser.add_argument('--image-path', type=str, required=True,
                                 help='Image file or directory path')
    add_faces_parser.set_defaults(func=add_face_images)
    
    # Recognize image command
    recognize_parser = subparsers.add_parser('recognize', help='Recognize faces in image')
    recognize_parser.add_argument('--image-path', type=str, required=True,
                                 help='Image file path')
    recognize_parser.add_argument('--threshold', type=float, default=0.6,
                                 help='Recognition threshold')
    recognize_parser.add_argument('--top-k', type=int, default=5,
                                 help='Number of top matches to show')
    recognize_parser.set_defaults(func=recognize_image)
    
    # Webcam command
    webcam_parser = subparsers.add_parser('webcam', help='Start real-time webcam recognition')
    webcam_parser.add_argument('--camera', type=int, default=0,
                              help='Camera device ID')
    webcam_parser.add_argument('--threshold', type=float, default=0.7,
                              help='Recognition threshold')
    webcam_parser.set_defaults(func=start_webcam)
    
    # Add person via webcam command
    add_webcam_parser = subparsers.add_parser('add-person-webcam', 
                                             help='Add person using webcam')
    add_webcam_parser.add_argument('--name', type=str, required=True,
                                  help='Person name')
    add_webcam_parser.add_argument('--camera', type=int, default=0,
                                  help='Camera device ID')
    add_webcam_parser.add_argument('--samples', type=int, default=5,
                                  help='Number of face samples to capture')
    add_webcam_parser.set_defaults(func=add_person_webcam)
    
    # List persons command
    list_parser = subparsers.add_parser('list', help='List all persons')
    list_parser.set_defaults(func=list_persons)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    benchmark_parser.add_argument('--test-images', type=str, required=True,
                                 help='Directory with test images')
    benchmark_parser.add_argument('--output-dir', type=str, default='benchmark_results',
                                 help='Output directory for results')
    benchmark_parser.set_defaults(func=run_benchmark)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up database')
    cleanup_parser.add_argument('--days', type=int, default=30,
                               help='Delete results older than N days')
    cleanup_parser.set_defaults(func=cleanup_database)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute command
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\\nOperation cancelled by user.")
        except Exception as e:
            logger.error(f"Command failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()