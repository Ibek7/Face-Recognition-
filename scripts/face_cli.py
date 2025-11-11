#!/usr/bin/env python3
"""
Comprehensive CLI for face recognition system.
Provides unified interface for all face recognition operations using Click.
"""
import sys
import os
import logging
from pathlib import Path
import click

# Add src to path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from database import DatabaseManager
from realtime import RealTimeFaceRecognizer
from embeddings import FaceEmbeddingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Standard exit codes
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_USER_CANCELLED = 2
EXIT_INVALID_INPUT = 3
EXIT_NOT_FOUND = 4


@click.group()
@click.option('--database', default='face_recognition.db', help='Database file path.')
@click.option('--encoder', type=click.Choice(['simple', 'dlib']), default='simple', help='Face encoder type.')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging.')
@click.pass_context
def cli(ctx, database, encoder, verbose):
    """A comprehensive CLI for the Face Recognition System."""
    ctx.ensure_object(dict)
    ctx.obj['database'] = database
    ctx.obj['encoder'] = encoder
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

@cli.command()
@click.pass_context
def setup(ctx):
    """Initialize database and show setup information."""
    try:
        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")
        
        click.echo("Face Recognition Database Setup")
        click.echo("=" * 40)
        
        stats = db_manager.get_recognition_stats()
        click.echo(f"Persons: {stats['total_persons']}")
        click.echo(f"Embeddings: {stats['total_embeddings']}")
        click.echo(f"Recognitions: {stats['total_recognitions']}")
        
        if stats['embeddings_by_encoder']:
            click.echo("\nEmbeddings by encoder:")
            for encoder, count in stats['embeddings_by_encoder'].items():
                click.echo(f"  {encoder}: {count}")
        
        click.echo(f"\nDatabase location: {ctx.obj['database']}")
        click.echo("Database is ready for use!")
        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_GENERAL_ERROR)

@cli.command('add-person')
@click.option('--name', required=True, help='Person name')
@click.option('--description', default='', help='Person description')
@click.pass_context
def add_person(ctx, name, description):
    """Add a new person to the database."""
    try:
        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")
        
        existing = db_manager.get_person_by_name(name)
        if existing:
            click.echo(f"Person '{name}' already exists (ID: {existing.id})")
            sys.exit(EXIT_GENERAL_ERROR)
        
        person = db_manager.add_person(name, description)
        click.echo(f"Added person: {person.name} (ID: {person.id})")
        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_GENERAL_ERROR)

@cli.command('add-faces')
@click.option('--person-name', required=True, help='Person name')
@click.option('--image-path', required=True, type=click.Path(exists=True), help='Image file or directory path')
@click.pass_context
def add_faces(ctx, person_name, image_path):
    """Add face images for a person."""
    try:
        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")
        embedding_manager = FaceEmbeddingManager(encoder_type=ctx.obj['encoder'])
        
        person = db_manager.get_person_by_name(person_name)
        if not person:
            click.echo(f"Person '{person_name}' not found. Add them first.", err=True)
            sys.exit(EXIT_NOT_FOUND)
        
        image_path = Path(image_path)
        
        if image_path.is_file():
            images = [image_path]
        elif image_path.is_dir():
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            images = []
            for ext in extensions:
                images.extend(image_path.glob(ext))
        else:
            click.echo(f"Invalid path: {image_path}", err=True)
            sys.exit(EXIT_INVALID_INPUT)
        
        click.echo(f"Processing {len(images)} images for {person.name}...")
        
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
                        encoder_type=ctx.obj['encoder'],
                        source_type="image"
                    )
                    total_embeddings += 1
            except Exception as e:
                click.echo(f"Error processing {img_path}: {e}", err=True)
        
        click.echo(f"Added {total_embeddings} face embeddings for {person.name}")
        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_GENERAL_ERROR)

@cli.command()
@click.option('--image-path', required=True, type=click.Path(exists=True), help='Image file path')
@click.option('--threshold', default=0.6, help='Recognition threshold')
@click.option('--top-k', default=5, help='Number of top matches to show')
@click.pass_context
def recognize(ctx, image_path, threshold, top_k):
    """Recognize faces in an image."""
    try:
        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")
        embedding_manager = FaceEmbeddingManager(encoder_type=ctx.obj['encoder'])
        
        embedding_data = embedding_manager.generate_embeddings_from_image(image_path)
        
        if not embedding_data['embeddings']:
            click.echo(f"No faces found in {image_path}")
            sys.exit(EXIT_SUCCESS)
        
        click.echo(f"Found {len(embedding_data['embeddings'])} faces in image:\n")
        
        for i, emb_info in enumerate(embedding_data['embeddings']):
            click.echo(f"Face {i + 1}:")
            click.echo(f"  Quality score: {emb_info['quality_score']:.2f}")
            
            similar_faces = db_manager.search_similar_faces(
                emb_info['embedding'],
                threshold=threshold,
                top_k=top_k
            )
            
            if similar_faces:
                click.echo(f"  Top matches:")
                for j, (face_embedding, similarity) in enumerate(similar_faces):
                    person = db_manager.get_person(face_embedding.person_id)
                    person_name = person.name if person else "Unknown"
                    click.echo(f"    {j+1}. {person_name} (similarity: {similarity:.3f})")
            else:
                click.echo(f"  No matches found (threshold: {threshold})")
            click.echo()
        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_GENERAL_ERROR)

@cli.command()
@click.option('--camera', default=0, help='Camera device ID')
@click.option('--threshold', default=0.7, help='Recognition threshold')
@click.pass_context
def webcam(ctx, camera, threshold):
    """Start real-time webcam recognition."""
    try:
        recognizer = RealTimeFaceRecognizer(
            db_path=ctx.obj['database'],
            encoder_type=ctx.obj['encoder'],
            confidence_threshold=threshold
        )
        
        recognizer.start_webcam(camera_id=camera, display=True)
        sys.exit(EXIT_SUCCESS)
    except KeyboardInterrupt:
        click.echo("\nStopping webcam recognition...")
        sys.exit(EXIT_USER_CANCELLED)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_GENERAL_ERROR)

@cli.command('list')
@click.pass_context
def list_persons(ctx):
    """List all persons in the database."""
    try:
        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")
        
        persons = db_manager.list_persons()
        
        if not persons:
            click.echo("No persons found in database.")
            sys.exit(EXIT_SUCCESS)
        
        click.echo(f"Found {len(persons)} persons:\n")
        
        for person in persons:
            embeddings = db_manager.get_face_embeddings(person_id=person.id)
            click.echo(f"ID: {person.id}")
            click.echo(f"Name: {person.name}")
            click.echo(f"Description: {person.description or 'None'}")
            click.echo(f"Face embeddings: {len(embeddings)}")
            click.echo(f"Added: {person.created_at}")
            click.echo("-" * 40)
        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_GENERAL_ERROR)

@cli.command()
@click.option('--days', default=30, help='Delete results older than N days')
@click.pass_context
def cleanup(ctx, days):
    """Clean up old recognition results from database."""
    try:
        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")
        
        deleted_count = db_manager.cleanup_old_results(days)
        click.echo(f"Deleted {deleted_count} old recognition results (older than {days} days)")
        
        stats = db_manager.get_recognition_stats()
        click.echo(f"Current recognitions in database: {stats['total_recognitions']}")
        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_GENERAL_ERROR)

if __name__ == '__main__':
    try:
        cli(obj={})
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user.")
        sys.exit(EXIT_USER_CANCELLED)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(EXIT_GENERAL_ERROR)
