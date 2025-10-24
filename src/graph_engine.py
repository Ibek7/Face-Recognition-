# Graph Query Engine

import threading
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque

@dataclass
class Node:
    """Graph node."""
    node_id: str
    label: str
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.node_id,
            'label': self.label,
            'properties': self.properties
        }

@dataclass
class Edge:
    """Graph edge."""
    edge_id: str
    source_id: str
    target_id: str
    relationship: str
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.edge_id,
            'source': self.source_id,
            'target': self.target_id,
            'relationship': self.relationship,
            'properties': self.properties
        }

class Graph:
    """Graph data structure."""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.adjacency: Dict[str, List[str]] = {}
        self.lock = threading.RLock()
    
    def add_node(self, node_id: str, label: str = "",
                properties: Dict = None) -> Node:
        """Add node."""
        with self.lock:
            node = Node(node_id, label, properties or {})
            self.nodes[node_id] = node
            
            if node_id not in self.adjacency:
                self.adjacency[node_id] = []
            
            return node
    
    def add_edge(self, source_id: str, target_id: str,
                relationship: str, properties: Dict = None) -> Edge:
        """Add edge."""
        import uuid
        
        with self.lock:
            edge_id = str(uuid.uuid4())
            edge = Edge(edge_id, source_id, target_id, relationship, properties or {})
            
            self.edges[edge_id] = edge
            
            if source_id in self.adjacency:
                self.adjacency[source_id].append(target_id)
            
            return edge
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbor nodes."""
        with self.lock:
            return self.adjacency.get(node_id, [])
    
    def get_edges(self, source_id: str, target_id: str = None) -> List[Edge]:
        """Get edges from source."""
        with self.lock:
            result = []
            for edge in self.edges.values():
                if edge.source_id == source_id:
                    if target_id is None or edge.target_id == target_id:
                        result.append(edge)
            return result

class GraphTraversal:
    """Graph traversal algorithms."""
    
    @staticmethod
    def bfs(graph: Graph, start_node: str, max_depth: int = None) -> List[str]:
        """Breadth-first search."""
        visited = set()
        queue = deque([(start_node, 0)])
        order = []
        
        while queue:
            node_id, depth = queue.popleft()
            
            if node_id in visited:
                continue
            
            if max_depth and depth > max_depth:
                continue
            
            visited.add(node_id)
            order.append(node_id)
            
            for neighbor in graph.get_neighbors(node_id):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return order
    
    @staticmethod
    def dfs(graph: Graph, start_node: str, max_depth: int = None) -> List[str]:
        """Depth-first search."""
        visited = set()
        order = []
        
        def dfs_recursive(node_id: str, depth: int = 0):
            if node_id in visited:
                return
            
            if max_depth and depth > max_depth:
                return
            
            visited.add(node_id)
            order.append(node_id)
            
            for neighbor in graph.get_neighbors(node_id):
                dfs_recursive(neighbor, depth + 1)
        
        dfs_recursive(start_node)
        return order
    
    @staticmethod
    def find_path(graph: Graph, start: str, end: str) -> Optional[List[str]]:
        """Find shortest path."""
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            node_id, path = queue.popleft()
            
            if node_id == end:
                return path
            
            for neighbor in graph.get_neighbors(node_id):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    @staticmethod
    def find_cycles(graph: Graph) -> List[List[str]]:
        """Find cycles in graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs_cycle(node_id: str, path: List[str]):
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            for neighbor in graph.get_neighbors(node_id):
                if neighbor not in visited:
                    dfs_cycle(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found cycle
                    idx = path.index(neighbor)
                    cycles.append(path[idx:] + [neighbor])
            
            rec_stack.remove(node_id)
        
        for node_id in graph.nodes:
            if node_id not in visited:
                dfs_cycle(node_id, [])
        
        return cycles

class GraphQuery:
    """Query and pattern matching on graphs."""
    
    def __init__(self, graph: Graph):
        self.graph = graph
    
    def find_by_label(self, label: str) -> List[str]:
        """Find nodes by label."""
        return [nid for nid, node in self.graph.nodes.items()
                if node.label == label]
    
    def find_by_property(self, property_name: str,
                        property_value: Any) -> List[str]:
        """Find nodes by property."""
        return [nid for nid, node in self.graph.nodes.items()
                if node.properties.get(property_name) == property_value]
    
    def match_pattern(self, pattern: Dict) -> List[List[str]]:
        """Find nodes matching pattern."""
        results = []
        
        start_nodes = self.find_by_label(pattern.get('label', ''))
        
        for node_id in start_nodes:
            if self._matches_properties(node_id, pattern.get('properties', {})):
                results.append([node_id])
        
        return results
    
    def _matches_properties(self, node_id: str, properties: Dict) -> bool:
        """Check if node matches properties."""
        node = self.graph.nodes.get(node_id)
        if not node:
            return False
        
        for key, value in properties.items():
            if node.properties.get(key) != value:
                return False
        
        return True
    
    def get_connected_component(self, node_id: str) -> Set[str]:
        """Get connected component."""
        visited = set()
        queue = deque([node_id])
        
        while queue:
            current = queue.popleft()
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return visited

class GraphAnalytics:
    """Graph analytics and metrics."""
    
    @staticmethod
    def calculate_degree(graph: Graph) -> Dict[str, int]:
        """Calculate node degrees."""
        degrees = {}
        for node_id in graph.nodes:
            degrees[node_id] = len(graph.get_neighbors(node_id))
        return degrees
    
    @staticmethod
    def find_central_nodes(graph: Graph, top_k: int = 10) -> List[Tuple[str, int]]:
        """Find most central nodes."""
        degrees = GraphAnalytics.calculate_degree(graph)
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]
    
    @staticmethod
    def calculate_clustering_coefficient(graph: Graph,
                                        node_id: str) -> float:
        """Calculate clustering coefficient."""
        neighbors = graph.get_neighbors(node_id)
        
        if len(neighbors) < 2:
            return 0.0
        
        edges_between = 0
        for i, node_a in enumerate(neighbors):
            for node_b in neighbors[i + 1:]:
                if node_b in graph.get_neighbors(node_a):
                    edges_between += 1
        
        possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
        
        return edges_between / possible_edges if possible_edges > 0 else 0.0

# Example usage
if __name__ == "__main__":
    graph = Graph()
    
    # Add nodes
    graph.add_node("user1", "User")
    graph.add_node("user2", "User")
    graph.add_node("face1", "Face")
    
    # Add edges
    graph.add_edge("user1", "face1", "has_face")
    graph.add_edge("user1", "user2", "knows")
    
    # Traversal
    bfs_result = GraphTraversal.bfs(graph, "user1")
    print(f"BFS from user1: {bfs_result}")
    
    # Query
    query = GraphQuery(graph)
    user_nodes = query.find_by_label("User")
    print(f"User nodes: {user_nodes}")
    
    # Analytics
    degrees = GraphAnalytics.calculate_degree(graph)
    print(f"Degrees: {degrees}")
