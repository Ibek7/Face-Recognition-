#!/usr/bin/env python3#!/usr/bin/env python3#!/usr/bin/env python3

"""

Comprehensive CLI for face recognition system.""""""

Provides unified interface for all face recognition operations using Click.

"""Comprehensive CLI for face recognition system.Comprehensive CLI for face recognition system.



import sysProvides unified interface for all face recognition operations using Click.Provides unified interface for all face recognition operations.

import os

from pathlib import Path""""""

import logging

import click



# Add src to pathimport sysimport sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import osimport os

from database import DatabaseManager

from realtime import RealTimeFaceRecognizerfrom pathlib import Pathfrom pathlib import Path

from embeddings import FaceEmbeddingManager

import loggingimport logging

# Configure logging

logging.basicConfig(import clickimport click

    level=logging.INFO,

    format='[%(levelname)s] %(name)s: %(message)s'

)

logger = logging.getLogger(__name__)# Add src to path# Add src to path



# Exit codessys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

EXIT_SUCCESS = 0

EXIT_GENERAL_ERROR = 1

EXIT_USER_CANCELLED = 2

EXIT_INVALID_INPUT = 3from database import DatabaseManagerfrom database import DatabaseManager

EXIT_NOT_FOUND = 4

from realtime import RealTimeFaceRecognizerfrom realtime import RealTimeFaceRecognizer



@click.group()from embeddings import FaceEmbeddingManagerfrom embeddings import FaceEmbeddingManager

@click.option('--database', default='face_recognition.db', help='Database file path')

@click.option('--encoder', type=click.Choice(['simple', 'dlib']), default='simple', help='Face encoder type')

@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')

@click.pass_context# Configure logging# Configure logging

def cli(ctx, database, encoder, verbose):

    """Comprehensive Face Recognition System CLI"""logging.basicConfig(logging.basicConfig(

    ctx.ensure_object(dict)

    ctx.obj['database'] = database    level=logging.INFO,    level=logging.INFO,

    ctx.obj['encoder'] = encoder

        format='[%(levelname)s] %(name)s: %(message)s'    format='[%(levelname)s] %(name)s: %(message)s'

    if verbose:

        logging.getLogger().setLevel(logging.DEBUG)))



logger = logging.getLogger(__name__)logger = logging.getLogger(__name__)

@cli.command()

@click.pass_context

def setup(ctx):

    """Initialize database and show setup information."""# Exit codes# Exit codes

    try:

        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")EXIT_SUCCESS = 0EXIT_SUCCESS = 0

        

        click.echo("Face Recognition Database Setup")EXIT_GENERAL_ERROR = 1EXIT_GENERAL_ERROR = 1

        click.echo("=" * 40)

        EXIT_USER_CANCELLED = 2EXIT_USER_CANCELLED = 2

        # Show statistics

        stats = db_manager.get_recognition_stats()EXIT_INVALID_INPUT = 3EXIT_INVALID_INPUT = 3

        click.echo(f"Persons: {stats['total_persons']}")

        click.echo(f"Embeddings: {stats['total_embeddings']}")EXIT_NOT_FOUND = 4EXIT_NOT_FOUND = 4

        click.echo(f"Recognitions: {stats['total_recognitions']}")

        

        if stats['embeddings_by_encoder']:

            click.echo("\nEmbeddings by encoder:")@click.group()

            for encoder, count in stats['embeddings_by_encoder'].items():

                click.echo(f"  {encoder}: {count}")@click.group()@click.option('--database', default='face_recognition.db', help='Database file path')

        

        click.echo(f"\nDatabase location: {ctx.obj['database']}")@click.option('--database', default='face_recognition.db', help='Database file path')@click.option('--encoder', type=click.Choice(['simple', 'dlib']), default='simple', help='Face encoder type')

        click.echo("Database is ready for use!")

        sys.exit(EXIT_SUCCESS)@click.option('--encoder', type=click.Choice(['simple', 'dlib']), default='simple', help='Face encoder type')@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')

    except Exception as e:

        click.echo(f"Error: {e}", err=True)@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')@click.pass_context

        sys.exit(EXIT_GENERAL_ERROR)

@click.pass_contextdef cli(ctx, database, encoder, verbose):



@cli.command('add-person')def cli(ctx, database, encoder, verbose):    """Comprehensive Face Recognition System CLI"""

@click.option('--name', required=True, help='Person name')

@click.option('--description', default='', help='Person description')    """Comprehensive Face Recognition System CLI"""    ctx.ensure_object(dict)

@click.pass_context

def add_person(ctx, name, description):    ctx.ensure_object(dict)    ctx.obj['database'] = database

    """Add a new person to the database."""

    try:    ctx.obj['database'] = database    ctx.obj['encoder'] = encoder

        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")

            ctx.obj['encoder'] = encoder    

        # Check if person already exists

        existing = db_manager.get_person_by_name(name)        if verbose:

        if existing:

            click.echo(f"Person '{name}' already exists (ID: {existing.id})")    if verbose:        logging.getLogger().setLevel(logging.DEBUG)

            sys.exit(EXIT_GENERAL_ERROR)

                logging.getLogger().setLevel(logging.DEBUG)

        # Add new person

        person = db_manager.add_person(name, description)@cli.command()

        click.echo(f"Added person: {person.name} (ID: {person.id})")

        sys.exit(EXIT_SUCCESS)@click.pass_context

    except Exception as e:

        click.echo(f"Error: {e}", err=True)@cli.command()def setup(ctx):

        sys.exit(EXIT_GENERAL_ERROR)

@click.pass_context    """Initialize database and show setup information."""



@cli.command('add-faces')def setup(ctx):    try:

@click.option('--person-name', required=True, help='Person name')

@click.option('--image-path', required=True, type=click.Path(exists=True), help='Image file or directory path')    """Initialize database and show setup information."""        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")

@click.pass_context

def add_faces(ctx, person_name, image_path):    try:        

    """Add face images for a person."""

    try:        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")        click.echo("Face Recognition Database Setup")

        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")

        embedding_manager = FaceEmbeddingManager(encoder_type=ctx.obj['encoder'])                click.echo("=" * 40)

        

        # Get person        click.echo("Face Recognition Database Setup")        

        person = db_manager.get_person_by_name(person_name)

        if not person:        click.echo("=" * 40)        # Show statistics

            click.echo(f"Person '{person_name}' not found. Add them first.", err=True)

            sys.exit(EXIT_NOT_FOUND)                stats = db_manager.get_recognition_stats()

        

        # Process images        # Show statistics        click.echo(f"Persons: {stats['total_persons']}")

        image_path = Path(image_path)

                stats = db_manager.get_recognition_stats()        click.echo(f"Embeddings: {stats['total_embeddings']}")

        if image_path.is_file():

            images = [image_path]        click.echo(f"Persons: {stats['total_persons']}")        click.echo(f"Recognitions: {stats['total_recognitions']}")

        elif image_path.is_dir():

            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']        click.echo(f"Embeddings: {stats['total_embeddings']}")        

            images = []

            for ext in extensions:        click.echo(f"Recognitions: {stats['total_recognitions']}")        if stats['embeddings_by_encoder']:

                images.extend(image_path.glob(ext))

        else:                    click.echo("\nEmbeddings by encoder:")

            click.echo(f"Invalid path: {image_path}", err=True)

            sys.exit(EXIT_INVALID_INPUT)        if stats['embeddings_by_encoder']:            for encoder, count in stats['embeddings_by_encoder'].items():

        

        click.echo(f"Processing {len(images)} images for {person.name}...")            click.echo("\nEmbeddings by encoder:")                click.echo(f"  {encoder}: {count}")

        

        total_embeddings = 0            for encoder, count in stats['embeddings_by_encoder'].items():        

        for img_path in images:

            try:                click.echo(f"  {encoder}: {count}")        click.echo(f"\nDatabase location: {ctx.obj['database']}")

                embedding_data = embedding_manager.generate_embeddings_from_image(str(img_path))

                                click.echo("Database is ready for use!")

                for emb_info in embedding_data['embeddings']:

                    db_manager.add_face_embedding(        click.echo(f"\nDatabase location: {ctx.obj['database']}")        sys.exit(EXIT_SUCCESS)

                        embedding=emb_info['embedding'],

                        person_id=person.id,        click.echo("Database is ready for use!")    except Exception as e:

                        source_image_path=str(img_path),

                        quality_score=emb_info['quality_score'],        sys.exit(EXIT_SUCCESS)        click.echo(f"Error: {e}", err=True)

                        encoder_type=ctx.obj['encoder'],

                        source_type="image"    except Exception as e:        sys.exit(EXIT_GENERAL_ERROR)

                    )

                    total_embeddings += 1        click.echo(f"Error: {e}", err=True)

            except Exception as e:

                click.echo(f"Error processing {img_path}: {e}", err=True)        sys.exit(EXIT_GENERAL_ERROR)def add_person(args):

        

        click.echo(f"Added {total_embeddings} face embeddings for {person.name}")    """Add a new person to the database."""

        sys.exit(EXIT_SUCCESS)

    except Exception as e:    db_manager = DatabaseManager(f"sqlite:///{args.database}")

        click.echo(f"Error: {e}", err=True)

        sys.exit(EXIT_GENERAL_ERROR)@cli.command('add-person')    



@click.option('--name', required=True, help='Person name')    # Check if person already exists

@cli.command()

@click.option('--image-path', required=True, type=click.Path(exists=True), help='Image file path')@click.option('--description', default='', help='Person description')    existing = db_manager.get_person_by_name(args.name)

@click.option('--threshold', default=0.6, help='Recognition threshold')

@click.option('--top-k', default=5, help='Number of top matches to show')@click.pass_context    if existing:

@click.pass_context

def recognize(ctx, image_path, threshold, top_k):def add_person(ctx, name, description):        print(f"Person '{args.name}' already exists (ID: {existing.id})")

    """Recognize faces in an image."""

    try:    """Add a new person to the database."""        return

        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")

        embedding_manager = FaceEmbeddingManager(encoder_type=ctx.obj['encoder'])    try:    

        

        # Generate embeddings for query image        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")    # Add new person

        embedding_data = embedding_manager.generate_embeddings_from_image(image_path)

                    person = db_manager.add_person(args.name, args.description or "")

        if not embedding_data['embeddings']:

            click.echo(f"No faces found in {image_path}")        # Check if person already exists    print(f"Added person: {person.name} (ID: {person.id})")

            sys.exit(EXIT_SUCCESS)

                existing = db_manager.get_person_by_name(name)

        click.echo(f"Found {len(embedding_data['embeddings'])} faces in image:\n")

                if existing:def add_face_images(args):

        # Search for each face

        for i, emb_info in enumerate(embedding_data['embeddings']):            click.echo(f"Person '{name}' already exists (ID: {existing.id})")    """Add face images for a person."""

            click.echo(f"Face {i + 1}:")

            click.echo(f"  Quality score: {emb_info['quality_score']:.2f}")            sys.exit(EXIT_GENERAL_ERROR)    db_manager = DatabaseManager(f"sqlite:///{args.database}")

            

            # Search database            embedding_manager = FaceEmbeddingManager(encoder_type=args.encoder)

            similar_faces = db_manager.search_similar_faces(

                emb_info['embedding'],        # Add new person    

                threshold=threshold,

                top_k=top_k        person = db_manager.add_person(name, description)    # Get person

            )

                    click.echo(f"Added person: {person.name} (ID: {person.id})")    person = db_manager.get_person_by_name(args.person_name)

            if similar_faces:

                click.echo(f"  Top matches:")        sys.exit(EXIT_SUCCESS)    if not person:

                for j, (face_embedding, similarity) in enumerate(similar_faces):

                    person = db_manager.get_person(face_embedding.person_id)    except Exception as e:        print(f"Person '{args.person_name}' not found. Add them first.")

                    person_name = person.name if person else "Unknown"

                    click.echo(f"    {j+1}. {person_name} (similarity: {similarity:.3f})")        click.echo(f"Error: {e}", err=True)        return

            else:

                click.echo(f"  No matches found (threshold: {threshold})")        sys.exit(EXIT_GENERAL_ERROR)    

            click.echo()

        sys.exit(EXIT_SUCCESS)    # Process images

    except Exception as e:

        click.echo(f"Error: {e}", err=True)    image_path = Path(args.image_path)

        sys.exit(EXIT_GENERAL_ERROR)

@cli.command('add-faces')    



@cli.command()@click.option('--person-name', required=True, help='Person name')    if image_path.is_file():

@click.option('--camera', default=0, help='Camera device ID')

@click.option('--threshold', default=0.7, help='Recognition threshold')@click.option('--image-path', required=True, type=click.Path(exists=True), help='Image file or directory path')        # Single image

@click.pass_context

def webcam(ctx, camera, threshold):@click.pass_context        images = [image_path]

    """Start real-time webcam recognition."""

    try:def add_faces(ctx, person_name, image_path):    elif image_path.is_dir():

        recognizer = RealTimeFaceRecognizer(

            db_path=ctx.obj['database'],    """Add face images for a person."""        # Directory of images

            encoder_type=ctx.obj['encoder'],

            confidence_threshold=threshold    try:        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

        )

                db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")        images = []

        recognizer.start_webcam(camera_id=camera, display=True)

        sys.exit(EXIT_SUCCESS)        embedding_manager = FaceEmbeddingManager(encoder_type=ctx.obj['encoder'])        for ext in extensions:

    except KeyboardInterrupt:

        click.echo("\nStopping webcam recognition...")                    images.extend(image_path.glob(ext))

        sys.exit(EXIT_USER_CANCELLED)

    except Exception as e:        # Get person    else:

        click.echo(f"Error: {e}", err=True)

        sys.exit(EXIT_GENERAL_ERROR)        person = db_manager.get_person_by_name(person_name)        print(f"Invalid path: {args.image_path}")



        if not person:        return

@cli.command()

@click.pass_context            click.echo(f"Person '{person_name}' not found. Add them first.", err=True)    

def list(ctx):

    """List all persons in the database."""            sys.exit(EXIT_NOT_FOUND)    print(f"Processing {len(images)} images for {person.name}...")

    try:

        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")            

        

        persons = db_manager.list_persons()        # Process images    total_embeddings = 0

        

        if not persons:        image_path = Path(image_path)    for img_path in images:

            click.echo("No persons found in database.")

            sys.exit(EXIT_SUCCESS)                try:

        

        click.echo(f"Found {len(persons)} persons:\n")        if image_path.is_file():            embedding_data = embedding_manager.generate_embeddings_from_image(str(img_path))

        

        for person in persons:            images = [image_path]            

            embeddings = db_manager.get_face_embeddings(person_id=person.id)

            click.echo(f"ID: {person.id}")        elif image_path.is_dir():            for emb_info in embedding_data['embeddings']:

            click.echo(f"Name: {person.name}")

            click.echo(f"Description: {person.description or 'None'}")            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']                db_manager.add_face_embedding(

            click.echo(f"Face embeddings: {len(embeddings)}")

            click.echo(f"Added: {person.created_at}")            images = []                    embedding=emb_info['embedding'],

            click.echo("-" * 40)

        sys.exit(EXIT_SUCCESS)            for ext in extensions:                    person_id=person.id,

    except Exception as e:

        click.echo(f"Error: {e}", err=True)                images.extend(image_path.glob(ext))                    source_image_path=str(img_path),

        sys.exit(EXIT_GENERAL_ERROR)

        else:                    quality_score=emb_info['quality_score'],



@cli.command()            click.echo(f"Invalid path: {image_path}", err=True)                    encoder_type=args.encoder,

@click.option('--days', default=30, help='Delete results older than N days')

@click.pass_context            sys.exit(EXIT_INVALID_INPUT)                    source_type="image"

def cleanup(ctx, days):

    """Clean up old recognition results from database."""                        )

    try:

        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")        click.echo(f"Processing {len(images)} images for {person.name}...")                total_embeddings += 1

        

        deleted_count = db_manager.cleanup_old_results(days)                    

        click.echo(f"Deleted {deleted_count} old recognition results (older than {days} days)")

                total_embeddings = 0        except Exception as e:

        # Show updated stats

        stats = db_manager.get_recognition_stats()        for img_path in images:            print(f"Error processing {img_path}: {e}")

        click.echo(f"Current recognitions in database: {stats['total_recognitions']}")

        sys.exit(EXIT_SUCCESS)            try:    

    except Exception as e:

        click.echo(f"Error: {e}", err=True)                embedding_data = embedding_manager.generate_embeddings_from_image(str(img_path))    print(f"Added {total_embeddings} face embeddings for {person.name}")

        sys.exit(EXIT_GENERAL_ERROR)

                



if __name__ == '__main__':                for emb_info in embedding_data['embeddings']:def recognize_image(args):

    try:

        cli(obj={})                    db_manager.add_face_embedding(    """Recognize faces in an image."""

    except KeyboardInterrupt:

        click.echo("\nOperation cancelled by user.")                        embedding=emb_info['embedding'],    db_manager = DatabaseManager(f"sqlite:///{args.database}")

        sys.exit(EXIT_USER_CANCELLED)

    except Exception as e:                        person_id=person.id,    embedding_manager = FaceEmbeddingManager(encoder_type=args.encoder)

        logger.error(f"Unexpected error: {e}")

        sys.exit(EXIT_GENERAL_ERROR)                        source_image_path=str(img_path),    


                        quality_score=emb_info['quality_score'],    try:

                        encoder_type=ctx.obj['encoder'],        # Generate embeddings for query image

                        source_type="image"        embedding_data = embedding_manager.generate_embeddings_from_image(args.image_path)

                    )        

                    total_embeddings += 1        if not embedding_data['embeddings']:

            except Exception as e:            print(f"No faces found in {args.image_path}")

                click.echo(f"Error processing {img_path}: {e}", err=True)            return

                

        click.echo(f"Added {total_embeddings} face embeddings for {person.name}")        print(f"Found {len(embedding_data['embeddings'])} faces in image:")

        sys.exit(EXIT_SUCCESS)        print()

    except Exception as e:        

        click.echo(f"Error: {e}", err=True)        # Search for each face

        sys.exit(EXIT_GENERAL_ERROR)        for i, emb_info in enumerate(embedding_data['embeddings']):

            print(f"Face {i + 1}:")

            print(f"  Quality score: {emb_info['quality_score']:.2f}")

@cli.command()            

@click.option('--image-path', required=True, type=click.Path(exists=True), help='Image file path')            # Search database

@click.option('--threshold', default=0.6, help='Recognition threshold')            similar_faces = db_manager.search_similar_faces(

@click.option('--top-k', default=5, help='Number of top matches to show')                emb_info['embedding'],

@click.pass_context                threshold=args.threshold,

def recognize(ctx, image_path, threshold, top_k):                top_k=args.top_k

    """Recognize faces in an image."""            )

    try:            

        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")            if similar_faces:

        embedding_manager = FaceEmbeddingManager(encoder_type=ctx.obj['encoder'])                print(f"  Top matches:")

                        for j, (face_embedding, similarity) in enumerate(similar_faces):

        # Generate embeddings for query image                    person = db_manager.get_person(face_embedding.person_id)

        embedding_data = embedding_manager.generate_embeddings_from_image(image_path)                    person_name = person.name if person else "Unknown"

                            print(f"    {j+1}. {person_name} (similarity: {similarity:.3f})")

        if not embedding_data['embeddings']:            else:

            click.echo(f"No faces found in {image_path}")                print(f"  No matches found (threshold: {args.threshold})")

            sys.exit(EXIT_SUCCESS)            print()

            

        click.echo(f"Found {len(embedding_data['embeddings'])} faces in image:\n")    except Exception as e:

                print(f"Error processing image: {e}")

        # Search for each face

        for i, emb_info in enumerate(embedding_data['embeddings']):def start_webcam(args):

            click.echo(f"Face {i + 1}:")    """Start real-time webcam recognition."""

            click.echo(f"  Quality score: {emb_info['quality_score']:.2f}")    recognizer = RealTimeFaceRecognizer(

                    db_path=args.database,

            # Search database        encoder_type=args.encoder,

            similar_faces = db_manager.search_similar_faces(        confidence_threshold=args.threshold

                emb_info['embedding'],    )

                threshold=threshold,    

                top_k=top_k    try:

            )        recognizer.start_webcam(camera_id=args.camera, display=True)

                except KeyboardInterrupt:

            if similar_faces:        print("\\nStopping webcam recognition...")

                click.echo(f"  Top matches:")    except Exception as e:

                for j, (face_embedding, similarity) in enumerate(similar_faces):        print(f"Error starting webcam: {e}")

                    person = db_manager.get_person(face_embedding.person_id)

                    person_name = person.name if person else "Unknown"def add_person_webcam(args):

                    click.echo(f"    {j+1}. {person_name} (similarity: {similarity:.3f})")    """Add a person using webcam capture."""

            else:    recognizer = RealTimeFaceRecognizer(

                click.echo(f"  No matches found (threshold: {threshold})")        db_path=args.database,

            click.echo()        encoder_type=args.encoder

        sys.exit(EXIT_SUCCESS)    )

    except Exception as e:    

        click.echo(f"Error: {e}", err=True)    success = recognizer.add_person_from_webcam(

        sys.exit(EXIT_GENERAL_ERROR)        person_name=args.name,

        camera_id=args.camera,

        num_samples=args.samples

@cli.command()    )

@click.option('--camera', default=0, help='Camera device ID')    

@click.option('--threshold', default=0.7, help='Recognition threshold')    if success:

@click.pass_context        print(f"Successfully added person: {args.name}")

def webcam(ctx, camera, threshold):    else:

    """Start real-time webcam recognition."""        print(f"Failed to add person: {args.name}")

    try:

        recognizer = RealTimeFaceRecognizer(def list_persons(args):

            db_path=ctx.obj['database'],    """List all persons in the database."""

            encoder_type=ctx.obj['encoder'],    db_manager = DatabaseManager(f"sqlite:///{args.database}")

            confidence_threshold=threshold    

        )    persons = db_manager.list_persons()

            

        recognizer.start_webcam(camera_id=camera, display=True)    if not persons:

        sys.exit(EXIT_SUCCESS)        print("No persons found in database.")

    except KeyboardInterrupt:        return

        click.echo("\nStopping webcam recognition...")    

        sys.exit(EXIT_USER_CANCELLED)    print(f"Found {len(persons)} persons:")

    except Exception as e:    print()

        click.echo(f"Error: {e}", err=True)    

        sys.exit(EXIT_GENERAL_ERROR)    for person in persons:

        embeddings = db_manager.get_face_embeddings(person_id=person.id)

        print(f"ID: {person.id}")

@cli.command()        print(f"Name: {person.name}")

@click.pass_context        print(f"Description: {person.description or 'None'}")

def list(ctx):        print(f"Face embeddings: {len(embeddings)}")

    """List all persons in the database."""        print(f"Added: {person.created_at}")

    try:        print("-" * 40)

        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")

        def run_benchmark(args):

        persons = db_manager.list_persons()    """Run performance benchmark."""

            from benchmark import FaceRecognitionBenchmark

        if not persons:    

            click.echo("No persons found in database.")    if not Path(args.test_images).exists():

            sys.exit(EXIT_SUCCESS)        print(f"Test images directory not found: {args.test_images}")

                return

        click.echo(f"Found {len(persons)} persons:\n")    

            benchmark = FaceRecognitionBenchmark(args.output_dir)

        for person in persons:    

            embeddings = db_manager.get_face_embeddings(person_id=person.id)    results = {}

            click.echo(f"ID: {person.id}")    

            click.echo(f"Name: {person.name}")    # Benchmark encoders

            click.echo(f"Description: {person.description or 'None'}")    print("Running encoder benchmark...")

            click.echo(f"Face embeddings: {len(embeddings)}")    encoder_results = benchmark.benchmark_encoders(args.test_images)

            click.echo(f"Added: {person.created_at}")    if encoder_results:

            click.echo("-" * 40)        results['encoders'] = encoder_results

        sys.exit(EXIT_SUCCESS)    

    except Exception as e:    # Generate report

        click.echo(f"Error: {e}", err=True)    if results:

        sys.exit(EXIT_GENERAL_ERROR)        benchmark.generate_report(results)

        print(f"Benchmark results saved to: {args.output_dir}")

    else:

@cli.command()        print("No benchmark results generated.")

@click.option('--days', default=30, help='Delete results older than N days')

@click.pass_contextdef cleanup_database(args):

def cleanup(ctx, days):    """Clean up old recognition results."""

    """Clean up old recognition results from database."""    db_manager = DatabaseManager(f"sqlite:///{args.database}")

    try:    

        db_manager = DatabaseManager(f"sqlite:///{ctx.obj['database']}")    if args.days:

                deleted_count = db_manager.cleanup_old_results(args.days)

        deleted_count = db_manager.cleanup_old_results(days)        print(f"Deleted {deleted_count} old recognition results (older than {args.days} days)")

        click.echo(f"Deleted {deleted_count} old recognition results (older than {days} days)")    

            # Show updated stats

        # Show updated stats    stats = db_manager.get_recognition_stats()

        stats = db_manager.get_recognition_stats()    print(f"Current recognitions in database: {stats['total_recognitions']}")

        click.echo(f"Current recognitions in database: {stats['total_recognitions']}")

        sys.exit(EXIT_SUCCESS)def main():

    except Exception as e:    parser = argparse.ArgumentParser(

        click.echo(f"Error: {e}", err=True)        description='Comprehensive Face Recognition System CLI',

        sys.exit(EXIT_GENERAL_ERROR)        formatter_class=argparse.RawDescriptionHelpFormatter,

        epilog="""

Examples:

if __name__ == '__main__':  %(prog)s setup --database faces.db

    try:  %(prog)s add-person --name "John Doe" --description "Employee"

        cli(obj={})  %(prog)s add-faces --person-name "John Doe" --image-path ./john_photos/

    except KeyboardInterrupt:  %(prog)s recognize --image-path photo.jpg --threshold 0.7

        click.echo("\nOperation cancelled by user.")  %(prog)s webcam --camera 0 --threshold 0.6

        sys.exit(EXIT_USER_CANCELLED)  %(prog)s add-person-webcam --name "Jane Doe" --samples 5

    except Exception as e:  %(prog)s benchmark --test-images ./test_data/

        logger.error(f"Unexpected error: {e}")        """

        sys.exit(EXIT_GENERAL_ERROR)    )

    
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