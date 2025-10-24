# Data Partitioning & Sharding System

import threading
import hashlib
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
import math

@dataclass
class Partition:
    """Data partition."""
    shard_id: int
    data: Dict[str, Any]
    
    def size(self) -> int:
        """Get partition size."""
        return len(self.data)

class ConsistentHashRing:
    """Consistent hashing for distributed partitioning."""
    
    def __init__(self, virtual_nodes: int = 160):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes: List[str] = []
        self.lock = threading.RLock()
    
    def add_node(self, node: str) -> None:
        """Add node to ring."""
        with self.lock:
            self.nodes.append(node)
            
            for i in range(self.virtual_nodes):
                virtual_key = f"{node}:{i}"
                hash_value = self._hash(virtual_key)
                self.ring[hash_value] = node
            
            self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node: str) -> None:
        """Remove node from ring."""
        with self.lock:
            self.nodes.remove(node)
            
            keys_to_remove = []
            for hash_value, n in self.ring.items():
                if n == node:
                    keys_to_remove.append(hash_value)
            
            for key in keys_to_remove:
                del self.ring[key]
            
            self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        """Get node for key."""
        with self.lock:
            if not self.ring:
                return None
            
            hash_value = self._hash(key)
            
            # Find next node
            for ring_key in self.sorted_keys:
                if ring_key >= hash_value:
                    return self.ring[ring_key]
            
            return self.ring[self.sorted_keys[0]]
    
    def _hash(self, key: str) -> int:
        """Hash key to ring position."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def get_replicas(self, key: str, replica_count: int = 3) -> List[str]:
        """Get replica nodes for key."""
        replicas = []
        with self.lock:
            hash_value = self._hash(key)
            
            for ring_key in self.sorted_keys:
                if ring_key >= hash_value:
                    node = self.ring[ring_key]
                    if node not in replicas:
                        replicas.append(node)
                    
                    if len(replicas) == replica_count:
                        break
            
            # Wrap around
            for ring_key in self.sorted_keys:
                if len(replicas) == replica_count:
                    break
                node = self.ring[ring_key]
                if node not in replicas:
                    replicas.append(node)
        
        return replicas

class RangePartitioner:
    """Range-based partitioning."""
    
    def __init__(self, partition_count: int, key_range: Tuple = None):
        self.partition_count = partition_count
        self.key_range = key_range or (0, 1000000)
        self.partitions: Dict[int, Partition] = {
            i: Partition(i, {}) for i in range(partition_count)
        }
        self.lock = threading.RLock()
    
    def get_partition_id(self, key: Any) -> int:
        """Get partition ID for key."""
        # Hash the key to a number
        hash_value = int(hashlib.md5(str(key).encode()).hexdigest(), 16)
        
        range_size = (self.key_range[1] - self.key_range[0]) / self.partition_count
        partition_id = int((hash_value % 1000000) / range_size)
        
        return min(partition_id, self.partition_count - 1)
    
    def put(self, key: str, value: Any) -> None:
        """Put value in partition."""
        partition_id = self.get_partition_id(key)
        
        with self.lock:
            self.partitions[partition_id].data[key] = value
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from partition."""
        partition_id = self.get_partition_id(key)
        
        with self.lock:
            partition = self.partitions[partition_id]
            return partition.data.get(key)
    
    def get_partition(self, partition_id: int) -> Optional[Partition]:
        """Get partition."""
        with self.lock:
            return self.partitions.get(partition_id)

class ShardingManager:
    """Manage data sharding across nodes."""
    
    def __init__(self, shard_count: int, hash_ring: Optional[ConsistentHashRing] = None):
        self.shard_count = shard_count
        self.hash_ring = hash_ring or ConsistentHashRing()
        self.local_data: Dict[str, Partition] = {
            str(i): Partition(i, {}) for i in range(shard_count)
        }
        self.lock = threading.RLock()
    
    def put(self, key: str, value: Any, replication: bool = False) -> Tuple[bool, List[str]]:
        """Put value with optional replication."""
        nodes = [self.hash_ring.get_node(key)]
        
        if replication:
            nodes = self.hash_ring.get_replicas(key)
        
        with self.lock:
            # Determine local shard
            shard_id = int(hashlib.md5(key.encode()).hexdigest(), 16) % self.shard_count
            shard_key = str(shard_id)
            
            if shard_key in self.local_data:
                self.local_data[shard_key].data[key] = value
                return True, nodes
        
        return False, nodes
    
    def get(self, key: str) -> Optional[Any]:
        """Get value."""
        with self.lock:
            shard_id = int(hashlib.md5(key.encode()).hexdigest(), 16) % self.shard_count
            shard_key = str(shard_id)
            
            if shard_key in self.local_data:
                return self.local_data[shard_key].data.get(key)
        
        return None
    
    def get_shard_distribution(self) -> Dict[str, int]:
        """Get distribution of data across shards."""
        with self.lock:
            return {
                shard_id: partition.size()
                for shard_id, partition in self.local_data.items()
            }
    
    def rebalance(self) -> Dict[str, List[Tuple[str, Any]]]:
        """Rebalance data across shards."""
        with self.lock:
            avg_size = sum(p.size() for p in self.local_data.values()) // len(self.local_data)
            moves = {}
            
            for shard_id, partition in self.local_data.items():
                if partition.size() > avg_size * 1.5:
                    overage = partition.size() - avg_size
                    moves[shard_id] = []
                    
                    items_to_move = []
                    for key, value in list(partition.data.items())[:overage]:
                        items_to_move.append((key, value))
                        del partition.data[key]
                    
                    moves[shard_id] = items_to_move
            
            return moves

class LocalityAware Partitioner:
    """Partition based on data locality."""
    
    def __init__(self):
        self.partitions: Dict[str, Dict[str, Any]] = {}
        self.locality_map: Dict[str, List[str]] = {}
        self.lock = threading.RLock()
    
    def create_partition(self, partition_id: str, location: str = "default") -> None:
        """Create partition with location."""
        with self.lock:
            self.partitions[partition_id] = {}
            self.locality_map[partition_id] = [location]
    
    def put(self, partition_id: str, key: str, value: Any) -> None:
        """Put value in partition."""
        with self.lock:
            if partition_id in self.partitions:
                self.partitions[partition_id][key] = value
    
    def get(self, partition_id: str, key: str) -> Optional[Any]:
        """Get value from partition."""
        with self.lock:
            if partition_id in self.partitions:
                return self.partitions[partition_id].get(key)
            return None
    
    def get_partition_location(self, partition_id: str) -> List[str]:
        """Get partition location."""
        with self.lock:
            return self.locality_map.get(partition_id, [])

# Example usage
if __name__ == "__main__":
    # Test consistent hashing
    ring = ConsistentHashRing()
    ring.add_node("node1")
    ring.add_node("node2")
    ring.add_node("node3")
    
    # Distribute keys
    for i in range(10):
        node = ring.get_node(f"key_{i}")
        print(f"key_{i} -> {node}")
    
    # Test range partitioning
    print("\nRange Partitioning:")
    partitioner = RangePartitioner(3)
    for i in range(6):
        partitioner.put(f"item_{i}", {"id": i})
    
    dist = {}
    for i in range(3):
        partition = partitioner.get_partition(i)
        dist[i] = partition.size() if partition else 0
    print(f"Distribution: {dist}")
    
    # Test sharding
    print("\nSharding:")
    sharding = ShardingManager(4, ring)
    for i in range(10):
        sharding.put(f"data_{i}", f"value_{i}")
    
    distribution = sharding.get_shard_distribution()
    print(f"Shard distribution: {distribution}")
