"""
Database manager for face recognition system.
Handles SQLAlchemy session management and database operations.
"""

import os
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
from functools import lru_cache

from src.models import Base, Person, FaceEmbedding, RecognitionResult, Dataset, ProcessingJob
from src.similarity import FaceSimilarity, DistanceMetric

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for face recognition system."""
    
    def __init__(self, db_url: str = "sqlite:///face_recognition.db"):
        """
        Initialize database manager.
        
        Args:
            db_url: SQLAlchemy database URL
        """
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
        
        # Initialize similarity calculator
        self.similarity_calc = FaceSimilarity(DistanceMetric.EUCLIDEAN)
        
        logger.info(f"Database initialized: {db_url}")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def add_person(self, name: str, description: str = "") -> Person:
        """
        Add a new person to the database.
        
        Args:
            name: Person's name
            description: Optional description
            
        Returns:
            Created Person object
        """
        with self.get_session() as session:
            person = Person(name=name, description=description)
            session.add(person)
            session.commit()
            session.refresh(person)
            return person
    
    def get_person(self, person_id: int) -> Optional[Person]:
        """Get person by ID."""
        with self.get_session() as session:
            return session.query(Person).filter(Person.id == person_id).first()

    @lru_cache(maxsize=128)
    def get_person_by_name(self, name: str) -> Optional[Person]:
        """Get person by name (cached)."""
        logger.info(f"Cache miss for get_person_by_name: {name}")
        with self.get_session() as session:
            return session.query(Person).filter(Person.name == name).first()
    
    def list_persons(self, active_only: bool = True) -> List[Person]:
        """List all persons in the database."""
        with self.get_session() as session:
            query = session.query(Person)
            if active_only:
                query = query.filter(Person.is_active == True)
            return query.all()

    def list_persons_paginated(
        self, page: int = 1, page_size: int = 10, active_only: bool = True
    ) -> Tuple[List[Person], int]:
        """List all persons in the database with pagination."""
        with self.get_session() as session:
            query = session.query(Person)
            if active_only:
                query = query.filter(Person.is_active == True)

            total = query.count()
            items = query.offset((page - 1) * page_size).limit(page_size).all()
            return items, total
    
    def add_face_embedding(self, 
                          embedding: np.ndarray,
                          person_id: Optional[int] = None,
                          source_image_path: str = "",
                          quality_score: float = 0.0,
                          encoder_type: str = "unknown",
                          **kwargs) -> FaceEmbedding:
        """
        Add a face embedding to the database.
        
        Args:
            embedding: Face embedding vector
            person_id: Optional person ID to associate with
            source_image_path: Path to source image
            quality_score: Face quality score
            encoder_type: Type of encoder used
            **kwargs: Additional metadata
            
        Returns:
            Created FaceEmbedding object
        """
        with self.get_session() as session:
            face_embedding = FaceEmbedding(
                person_id=person_id,
                source_image_path=source_image_path,
                quality_score=quality_score,
                encoder_type=encoder_type,
                source_type=kwargs.get('source_type', 'image'),
                face_confidence=kwargs.get('face_confidence', 0.0),
                preprocessing_params=json.dumps(kwargs.get('preprocessing_params', {}))
            )
            face_embedding.set_embedding(embedding)
            
            session.add(face_embedding)
            session.commit()
            session.refresh(face_embedding)
            return face_embedding
    
    def get_face_embeddings(self, 
                           person_id: Optional[int] = None,
                           encoder_type: Optional[str] = None,
                           min_quality: float = 0.0) -> List[FaceEmbedding]:
        """
        Get face embeddings with optional filters.
        
        Args:
            person_id: Filter by person ID
            encoder_type: Filter by encoder type
            min_quality: Minimum quality score
            
        Returns:
            List of FaceEmbedding objects
        """
        with self.get_session() as session:
            query = session.query(FaceEmbedding)
            
            if person_id is not None:
                query = query.filter(FaceEmbedding.person_id == person_id)
            
            if encoder_type is not None:
                query = query.filter(FaceEmbedding.encoder_type == encoder_type)
            
            if min_quality > 0:
                query = query.filter(FaceEmbedding.quality_score >= min_quality)
            
            return query.all()
    
    def search_similar_faces(self, 
                           query_embedding: np.ndarray,
                           threshold: float = 0.5,
                           top_k: int = 10,
                           person_id: Optional[int] = None) -> List[Tuple[FaceEmbedding, float]]:
        """
        Search for similar faces in the database.
        
        Args:
            query_embedding: Query face embedding
            threshold: Similarity threshold
            top_k: Maximum number of results
            person_id: Optional person ID filter
            
        Returns:
            List of (FaceEmbedding, similarity_score) tuples
        """
        with self.get_session() as session:
            query = session.query(FaceEmbedding)
            
            if person_id is not None:
                query = query.filter(FaceEmbedding.person_id == person_id)
            
            embeddings = query.all()
            
            # Calculate similarities
            results = []
            for embedding_obj in embeddings:
                stored_embedding = embedding_obj.get_embedding()
                similarity = self.similarity_calc.calculate_similarity(query_embedding, stored_embedding)
                
                if similarity >= threshold:
                    results.append((embedding_obj, similarity))
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def add_recognition_result(self,
                             person_id: Optional[int],
                             confidence_score: float,
                             source_image_path: str = "",
                             processing_time: float = 0.0,
                             **kwargs) -> RecognitionResult:
        """
        Add a recognition result to the database.
        
        Args:
            person_id: Recognized person ID (None for unknown)
            confidence_score: Recognition confidence
            source_image_path: Source image path
            processing_time: Processing time in seconds
            **kwargs: Additional metadata
            
        Returns:
            Created RecognitionResult object
        """
        with self.get_session() as session:
            result = RecognitionResult(
                person_id=person_id,
                confidence_score=confidence_score,
                source_image_path=source_image_path,
                processing_time=processing_time,
                distance_score=kwargs.get('distance_score', 0.0),
                similarity_metric=kwargs.get('similarity_metric', 'euclidean'),
                source_type=kwargs.get('source_type', 'image'),
                detection_bbox=json.dumps(kwargs.get('detection_bbox', [])),
                model_version=kwargs.get('model_version', 'unknown')
            )
            
            session.add(result)
            session.commit()
            session.refresh(result)
            return result
    
    def get_recognition_stats(self) -> Dict[str, Any]:
        """Get recognition system statistics."""
        with self.get_session() as session:
            stats = {}
            
            # Person counts
            stats['total_persons'] = session.query(Person).filter(Person.is_active == True).count()
            
            # Embedding counts
            stats['total_embeddings'] = session.query(FaceEmbedding).count()
            stats['embeddings_by_encoder'] = {}
            
            encoder_counts = session.query(
                FaceEmbedding.encoder_type,
                session.query(FaceEmbedding).filter(
                    FaceEmbedding.encoder_type == FaceEmbedding.encoder_type
                ).count().label('count')
            ).group_by(FaceEmbedding.encoder_type).all()
            
            for encoder_type, count in encoder_counts:
                stats['embeddings_by_encoder'][encoder_type] = count
            
            # Recognition counts
            stats['total_recognitions'] = session.query(RecognitionResult).count()
            stats['successful_recognitions'] = session.query(RecognitionResult).filter(
                RecognitionResult.person_id.isnot(None)
            ).count()
            
            # Average processing time
            avg_time = session.query(
                session.query(RecognitionResult.processing_time).filter(
                    RecognitionResult.processing_time > 0
                ).avg()
            ).scalar()
            stats['avg_processing_time'] = float(avg_time) if avg_time else 0.0
            
            return stats
    
    def cleanup_old_results(self, days: int = 30) -> int:
        """
        Clean up old recognition results.
        
        Args:
            days: Keep results from last N days
            
        Returns:
            Number of deleted records
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self.get_session() as session:
            deleted_count = session.query(RecognitionResult).filter(
                RecognitionResult.recognized_at < cutoff_date
            ).delete()
            
            session.commit()
            logger.info(f"Cleaned up {deleted_count} old recognition results")
            return deleted_count