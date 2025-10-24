# Vector Database

import threading
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import math

@dataclass
class Vector:
    """Vector for similarity search."""
    vector_id: str
    embedding: List[float]
    metadata: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

class VectorDistance:
    """Vector distance calculations."""
    
    @staticmethod
    def euclidean_distance(v1: List[float], v2: List[float]) -> float:
        """Calculate Euclidean distance."""
        if len(v1) != len(v2):
            return float('inf')
        
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
    
    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity."""
        if len(v1) != len(v2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = math.sqrt(sum(a ** 2 for a in v1))
        norm_v2 = math.sqrt(sum(b ** 2 for b in v2))
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)
    
    @staticmethod
    def manhattan_distance(v1: List[float], v2: List[float]) -> float:
        """Calculate Manhattan distance."""
        if len(v1) != len(v2):
            return float('inf')
        
        return sum(abs(a - b) for a, b in zip(v1, v2))

class HNSW:
    """Hierarchical Navigable Small World (HNSW) for approximate nearest neighbors."""
    
    def __init__(self, max_connections: int = 16, ef_construction: int = 200):
        self.max_connections = max_connections
        self.ef_construction = ef_construction
        self.vectors: Dict[str, Vector] = {}
        self.graph: Dict[str, List[str]] = {}
        self.entry_point: Optional[str] = None
        self.lock = threading.RLock()
    
    def insert(self, vector: Vector) -> None:
        """Insert vector into index."""
        with self.lock:
            if not self.vectors:
                self.entry_point = vector.vector_id
            
            self.vectors[vector.vector_id] = vector
            self.graph[vector.vector_id] = []
            
            # Connect to nearest neighbors
            if self.entry_point and vector.vector_id != self.entry_point:
                neighbors = self._find_nearest_neighbors(
                    vector.embedding,
                    self.max_connections
                )
                
                for neighbor_id in neighbors:
                    self.graph[vector.vector_id].append(neighbor_id)
                    if neighbor_id not in self.graph[vector.vector_id]:
                        self.graph[neighbor_id].append(vector.vector_id)
    
    def search_knn(self, query_embedding: List[float], k: int = 10) -> List[Tuple[str, float]]:
        """K-nearest neighbors search."""
        with self.lock:
            if not self.vectors or not self.entry_point:
                return []
            
            # Start from entry point
            visited = {self.entry_point}
            candidates = [(self._distance(query_embedding, 
                                         self.vectors[self.entry_point].embedding),
                          self.entry_point)]
            w = []
            
            # Greedy search
            for _ in range(self.ef_construction):
                if not candidates:
                    break
                
                dist, current = min(candidates)
                candidates.remove((dist, current))
                
                if dist > max((d, v) for d, v in w)[0] if w else float('inf'):
                    break
                
                # Check neighbors
                for neighbor_id in self.graph.get(current, []):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        neighbor_vec = self.vectors[neighbor_id].embedding
                        neighbor_dist = self._distance(query_embedding, neighbor_vec)
                        
                        if neighbor_dist < max((d, v) for d, v in w)[0] if w else float('inf'):
                            candidates.append((neighbor_dist, neighbor_id))
                            w.append((neighbor_dist, neighbor_id))
                            
                            if len(w) > k:
                                w.sort(reverse=True)
                                w.pop(0)
            
            w.sort()
            return [(vid, dist) for dist, vid in w[:k]]
    
    def _find_nearest_neighbors(self, embedding: List[float],
                               k: int) -> List[str]:
        """Find k nearest neighbors."""
        distances = []
        
        for vid, vector in self.vectors.items():
            dist = self._distance(embedding, vector.embedding)
            distances.append((dist, vid))
        
        distances.sort()
        return [vid for _, vid in distances[:k]]
    
    def _distance(self, v1: List[float], v2: List[float]) -> float:
        """Calculate distance between vectors."""
        return VectorDistance.euclidean_distance(v1, v2)

class VectorDB:
    """Vector database for similarity search."""
    
    def __init__(self, distance_metric: str = "cosine"):
        self.vectors: Dict[str, Vector] = {}
        self.indexes: Dict[str, HNSW] = {}
        self.distance_metric = distance_metric
        self.lock = threading.RLock()
    
    def add_vector(self, vector_id: str, embedding: List[float],
                  metadata: Dict = None) -> None:
        """Add vector to database."""
        metadata = metadata or {}
        
        with self.lock:
            vector = Vector(vector_id, embedding, metadata)
            self.vectors[vector_id] = vector
            
            # Create index if doesn't exist
            if "default" not in self.indexes:
                self.indexes["default"] = HNSW()
            
            self.indexes["default"].insert(vector)
    
    def search(self, query_embedding: List[float], k: int = 10,
              index_name: str = "default") -> List[Dict]:
        """Search for similar vectors."""
        with self.lock:
            if index_name not in self.indexes:
                return []
            
            results = self.indexes[index_name].search_knn(query_embedding, k)
            
            output = []
            for vector_id, distance in results:
                if vector_id in self.vectors:
                    vector = self.vectors[vector_id]
                    
                    # Convert distance to similarity based on metric
                    if self.distance_metric == "cosine":
                        similarity = distance  # Already cosine similarity
                    else:
                        similarity = 1.0 / (1.0 + distance)
                    
                    output.append({
                        'vector_id': vector_id,
                        'distance': distance,
                        'similarity': similarity,
                        'metadata': vector.metadata
                    })
            
            return output
    
    def delete_vector(self, vector_id: str) -> None:
        """Delete vector from database."""
        with self.lock:
            if vector_id in self.vectors:
                del self.vectors[vector_id]
    
    def batch_add_vectors(self, vectors: List[Tuple[str, List[float], Dict]]) -> None:
        """Add multiple vectors."""
        with self.lock:
            for vector_id, embedding, metadata in vectors:
                self.add_vector(vector_id, embedding, metadata)
    
    def get_vector(self, vector_id: str) -> Optional[Vector]:
        """Get vector by ID."""
        with self.lock:
            return self.vectors.get(vector_id)
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self.lock:
            return {
                'total_vectors': len(self.vectors),
                'indexes': len(self.indexes),
                'distance_metric': self.distance_metric
            }

class FaceEmbeddingDB:
    """Specialized vector DB for face embeddings."""
    
    def __init__(self):
        self.db = VectorDB(distance_metric="cosine")
        self.face_metadata: Dict[str, Dict] = {}
        self.lock = threading.RLock()
    
    def add_face(self, face_id: str, embedding: List[float],
                person_id: str = None, source: str = None) -> None:
        """Add face embedding."""
        metadata = {
            'person_id': person_id,
            'source': source,
            'type': 'face'
        }
        
        with self.lock:
            self.db.add_vector(face_id, embedding, metadata)
            self.face_metadata[face_id] = metadata
    
    def find_similar_faces(self, query_embedding: List[float],
                          threshold: float = 0.6, limit: int = 10) -> List[Dict]:
        """Find similar faces."""
        results = self.db.search(query_embedding, limit)
        
        # Filter by threshold
        filtered = [r for r in results if r['similarity'] >= threshold]
        
        return filtered
    
    def get_person_faces(self, person_id: str) -> List[Dict]:
        """Get all faces for a person."""
        with self.lock:
            faces = []
            for face_id, metadata in self.face_metadata.items():
                if metadata.get('person_id') == person_id:
                    vector = self.db.get_vector(face_id)
                    if vector:
                        faces.append({
                            'face_id': face_id,
                            'metadata': metadata
                        })
            
            return faces

# Example usage
if __name__ == "__main__":
    # Create vector DB
    db = VectorDB()
    
    # Add face embeddings (simulated)
    embeddings = [
        ("face-1", [0.1, 0.2, 0.3, 0.4]),
        ("face-2", [0.1, 0.2, 0.3, 0.45]),
        ("face-3", [0.9, 0.8, 0.7, 0.6]),
        ("face-4", [0.15, 0.25, 0.35, 0.45]),
    ]
    
    for face_id, embedding in embeddings:
        db.add_vector(face_id, embedding, {'type': 'face'})
    
    # Search for similar faces
    query_embedding = [0.1, 0.2, 0.3, 0.4]
    results = db.search(query_embedding, k=3)
    
    print("Similar faces:")
    for result in results:
        print(f"  {result['vector_id']}: similarity={result['similarity']:.4f}")
    
    # Get stats
    stats = db.get_stats()
    print(f"\nDB Stats: {json.dumps(stats, indent=2)}")
