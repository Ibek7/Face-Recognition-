"""
SQLAlchemy models for face recognition database.
Stores face embeddings, metadata, and recognition results.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, LargeBinary, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import numpy as np
import pickle

Base = declarative_base()

class Person(Base):
    """Represents a person in the face recognition system."""
    __tablename__ = 'persons'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationship to face embeddings
    face_embeddings = relationship("FaceEmbedding", back_populates="person")
    recognition_results = relationship("RecognitionResult", back_populates="person")

class FaceEmbedding(Base):
    """Stores face embeddings with metadata."""
    __tablename__ = 'face_embeddings'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id'), nullable=True)
    
    # Embedding data (stored as pickled numpy array)
    embedding_data = Column(LargeBinary, nullable=False)
    embedding_dim = Column(Integer, nullable=False)
    
    # Source information
    source_image_path = Column(String(500))
    source_type = Column(String(50))  # 'image', 'video', 'webcam', etc.
    
    # Quality metrics
    quality_score = Column(Float)
    face_confidence = Column(Float)
    
    # Processing metadata
    encoder_type = Column(String(50))
    preprocessing_params = Column(Text)  # JSON string of parameters
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    person = relationship("Person", back_populates="face_embeddings")
    
    def get_embedding(self) -> np.ndarray:
        """Deserialize and return the embedding as numpy array."""
        return pickle.loads(self.embedding_data)
    
    def set_embedding(self, embedding: np.ndarray) -> None:
        """Serialize and store the embedding."""
        self.embedding_data = pickle.dumps(embedding)
        self.embedding_dim = len(embedding)

class RecognitionResult(Base):
    """Stores face recognition results and predictions."""
    __tablename__ = 'recognition_results'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id'), nullable=True)
    
    # Recognition details
    confidence_score = Column(Float, nullable=False)
    distance_score = Column(Float)
    similarity_metric = Column(String(50))
    
    # Source of recognition
    source_image_path = Column(String(500))
    source_type = Column(String(50))
    detection_bbox = Column(String(100))  # JSON string: [x, y, width, height]
    
    # Processing info
    processing_time = Column(Float)  # Processing time in seconds
    model_version = Column(String(50))
    
    # Timestamps
    recognized_at = Column(DateTime, default=func.now())
    
    # Relationships
    person = relationship("Person", back_populates="recognition_results")

class Dataset(Base):
    """Represents a dataset or collection of images."""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    dataset_path = Column(String(500))
    
    # Statistics
    total_images = Column(Integer, default=0)
    total_faces = Column(Integer, default=0)
    processed_images = Column(Integer, default=0)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_started_at = Column(DateTime)
    processing_completed_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class ProcessingJob(Base):
    """Tracks batch processing jobs."""
    __tablename__ = 'processing_jobs'
    
    id = Column(Integer, primary_key=True)
    job_type = Column(String(50), nullable=False)  # 'embedding', 'recognition', 'benchmark'
    status = Column(String(20), default='pending')  # 'pending', 'running', 'completed', 'failed'
    
    # Job parameters
    input_path = Column(String(500))
    output_path = Column(String(500))
    parameters = Column(Text)  # JSON string of job parameters
    
    # Progress tracking
    total_items = Column(Integer, default=0)
    processed_items = Column(Integer, default=0)
    failed_items = Column(Integer, default=0)
    
    # Results
    result_summary = Column(Text)  # JSON string of results
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate job progress as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100