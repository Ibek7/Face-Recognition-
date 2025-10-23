# Advanced Data Lineage & Provenance Tracking

import threading
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

class DataOrigin(Enum):
    """Origin of data."""
    USER_INPUT = "user_input"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    MODEL_OUTPUT = "model_output"
    DERIVED = "derived"
    CACHED = "cached"

class TransformationType(Enum):
    """Type of data transformation."""
    FILTER = "filter"
    MAP = "map"
    AGGREGATE = "aggregate"
    JOIN = "join"
    ENCODE = "encode"
    NORMALIZE = "normalize"
    AUGMENT = "augment"

@dataclass
class DataNode:
    """Node in data lineage graph."""
    node_id: str
    data_hash: str
    origin: DataOrigin
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'data_hash': self.data_hash,
            'origin': self.origin.value,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'size_bytes': self.size_bytes
        }

@dataclass
class DataTransformation:
    """Data transformation operation."""
    transform_id: str
    transformation_type: TransformationType
    input_nodes: List[str]
    output_node: str
    timestamp: float = field(default_factory=time.time)
    parameters: Dict[str, Any] = field(default_factory=dict)
    operator: str = ""
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'transform_id': self.transform_id,
            'type': self.transformation_type.value,
            'input_nodes': self.input_nodes,
            'output_node': self.output_node,
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'operator': self.operator,
            'duration_ms': self.duration_ms
        }

class LineageGraph:
    """Graph representing data lineage."""
    
    def __init__(self):
        self.nodes: Dict[str, DataNode] = {}
        self.transformations: Dict[str, DataTransformation] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)  # Parent -> children
        self.reverse_edges: Dict[str, List[str]] = defaultdict(list)  # Child -> parents
        self.lock = threading.RLock()
    
    def add_node(self, node: DataNode) -> None:
        """Add node to graph."""
        with self.lock:
            self.nodes[node.node_id] = node
    
    def add_transformation(self, transform: DataTransformation) -> None:
        """Add transformation."""
        with self.lock:
            self.transformations[transform.transform_id] = transform
            
            # Add edges
            for input_node in transform.input_nodes:
                self.edges[input_node].append(transform.output_node)
                self.reverse_edges[transform.output_node].append(input_node)
    
    def get_upstream_lineage(self, node_id: str, max_depth: int = 10) -> List[str]:
        """Get upstream data lineage."""
        with self.lock:
            visited = set()
            queue = deque([(node_id, 0)])
            lineage = []
            
            while queue:
                current, depth = queue.popleft()
                
                if current in visited or depth > max_depth:
                    continue
                
                visited.add(current)
                lineage.append(current)
                
                # Add parents
                for parent in self.reverse_edges.get(current, []):
                    queue.append((parent, depth + 1))
            
            return lineage
    
    def get_downstream_lineage(self, node_id: str, max_depth: int = 10) -> List[str]:
        """Get downstream data lineage."""
        with self.lock:
            visited = set()
            queue = deque([(node_id, 0)])
            lineage = []
            
            while queue:
                current, depth = queue.popleft()
                
                if current in visited or depth > max_depth:
                    continue
                
                visited.add(current)
                lineage.append(current)
                
                # Add children
                for child in self.edges.get(current, []):
                    queue.append((child, depth + 1))
            
            return lineage
    
    def find_common_ancestors(self, node_ids: List[str]) -> Set[str]:
        """Find common ancestors."""
        if not node_ids:
            return set()
        
        with self.lock:
            ancestors = set(self.get_upstream_lineage(node_ids[0]))
            
            for node_id in node_ids[1:]:
                node_ancestors = set(self.get_upstream_lineage(node_id))
                ancestors &= node_ancestors
            
            return ancestors
    
    def get_graph_json(self) -> Dict:
        """Export graph as JSON."""
        with self.lock:
            return {
                'nodes': {
                    node_id: node.to_dict()
                    for node_id, node in self.nodes.items()
                },
                'transformations': {
                    t_id: t.to_dict()
                    for t_id, t in self.transformations.items()
                }
            }

class ProvenanceTracker:
    """Track data provenance."""
    
    def __init__(self):
        self.lineage_graph = LineageGraph()
        self.provenance_records: Dict[str, Dict] = {}
        self.lock = threading.RLock()
    
    def track_data_origin(self, node_id: str, data: Any, origin: DataOrigin,
                         metadata: Dict = None) -> None:
        """Track data origin."""
        data_hash = self._compute_hash(data)
        node = DataNode(
            node_id=node_id,
            data_hash=data_hash,
            origin=origin,
            metadata=metadata or {},
            size_bytes=len(str(data).encode())
        )
        
        self.lineage_graph.add_node(node)
        
        with self.lock:
            self.provenance_records[node_id] = {
                'origin': origin.value,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
    
    def track_transformation(self, input_node_ids: List[str], output_node_id: str,
                           transform_type: TransformationType, 
                           parameters: Dict = None, duration_ms: float = 0.0) -> None:
        """Track data transformation."""
        transform_id = self._generate_id()
        
        transform = DataTransformation(
            transform_id=transform_id,
            transformation_type=transform_type,
            input_nodes=input_node_ids,
            output_node=output_node_id,
            parameters=parameters or {},
            duration_ms=duration_ms
        )
        
        self.lineage_graph.add_transformation(transform)
    
    def get_provenance(self, node_id: str) -> Dict:
        """Get complete provenance for node."""
        upstream = self.lineage_graph.get_upstream_lineage(node_id)
        
        with self.lock:
            return {
                'node_id': node_id,
                'provenance': self.provenance_records.get(node_id),
                'lineage': upstream,
                'graph': self.lineage_graph.get_graph_json()
            }
    
    def verify_data_integrity(self, node_id: str, data: Any) -> bool:
        """Verify data hasn't been tampered."""
        if node_id not in self.lineage_graph.nodes:
            return False
        
        current_hash = self._compute_hash(data)
        stored_hash = self.lineage_graph.nodes[node_id].data_hash
        
        return current_hash == stored_hash
    
    def _compute_hash(self, data: Any) -> str:
        """Compute data hash."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

class DataQualityValidator:
    """Validate data quality along lineage."""
    
    def __init__(self):
        self.quality_metrics: Dict[str, Dict] = {}
        self.lock = threading.RLock()
    
    def record_quality(self, node_id: str, completeness: float,
                      accuracy: float, consistency: float) -> None:
        """Record data quality metrics."""
        with self.lock:
            self.quality_metrics[node_id] = {
                'completeness': completeness,
                'accuracy': accuracy,
                'consistency': consistency,
                'timestamp': time.time(),
                'quality_score': (completeness + accuracy + consistency) / 3
            }
    
    def get_quality_report(self, node_ids: List[str]) -> Dict:
        """Get quality report for nodes."""
        with self.lock:
            report = {}
            
            for node_id in node_ids:
                if node_id in self.quality_metrics:
                    report[node_id] = self.quality_metrics[node_id]
            
            avg_quality = sum(m['quality_score'] for m in report.values()) / len(report) if report else 0
            
            return {
                'nodes': report,
                'average_quality': avg_quality
            }

class DataCatalog:
    """Catalog datasets with lineage."""
    
    def __init__(self):
        self.datasets: Dict[str, Dict] = {}
        self.tracker = ProvenanceTracker()
        self.lock = threading.RLock()
    
    def register_dataset(self, dataset_id: str, name: str, description: str,
                        owner: str, data: Any = None) -> None:
        """Register dataset in catalog."""
        with self.lock:
            self.datasets[dataset_id] = {
                'id': dataset_id,
                'name': name,
                'description': description,
                'owner': owner,
                'created_at': time.time(),
                'last_updated': time.time()
            }
        
        if data:
            self.tracker.track_data_origin(
                dataset_id, data, DataOrigin.DATABASE,
                {'name': name, 'owner': owner}
            )
    
    def get_dataset_lineage(self, dataset_id: str) -> Dict:
        """Get lineage for dataset."""
        return self.tracker.get_provenance(dataset_id)
    
    def list_datasets(self) -> List[Dict]:
        """List all datasets."""
        with self.lock:
            return list(self.datasets.values())

# Example usage
if __name__ == "__main__":
    # Create tracker
    tracker = ProvenanceTracker()
    
    # Track data origins
    tracker.track_data_origin(
        "raw_faces",
        {'images': ['img1.jpg', 'img2.jpg']},
        DataOrigin.USER_INPUT,
        {'source': 'webcam', 'timestamp': '2025-10-22'}
    )
    
    # Track transformation
    tracker.track_transformation(
        input_node_ids=['raw_faces'],
        output_node_id='preprocessed_faces',
        transform_type=TransformationType.NORMALIZE,
        parameters={'method': 'histogram_equalization'},
        duration_ms=145.2
    )
    
    # Track another transformation
    tracker.track_transformation(
        input_node_ids=['preprocessed_faces'],
        output_node_id='encoded_faces',
        transform_type=TransformationType.ENCODE,
        parameters={'model': 'facenet'},
        duration_ms=89.5
    )
    
    # Get provenance
    provenance = tracker.get_provenance('encoded_faces')
    print("Data Provenance:")
    print(json.dumps(provenance, indent=2, default=str))
