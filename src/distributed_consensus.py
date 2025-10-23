# Distributed System Consensus & Coordination

import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

class ConsensusAlgorithm(Enum):
    """Consensus algorithms."""
    RAFT = "raft"
    PAXOS = "paxos"
    VOTING = "voting"

class NodeState(Enum):
    """Node states in cluster."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

@dataclass
class LogEntry:
    """Replicated log entry."""
    index: int
    term: int
    command: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class RaftNode:
    """Raft consensus algorithm implementation."""
    
    def __init__(self, node_id: str, peer_ids: List[str]):
        self.node_id = node_id
        self.peer_ids = [p for p in peer_ids if p != node_id]
        
        # Persistent state
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        
        # Volatile state
        self.commit_index = 0
        self.last_applied = 0
        self.state = NodeState.FOLLOWER
        self.election_timeout = time.time() + 1.5
        
        # Leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        self.lock = threading.RLock()
    
    def append_entry(self, command: str, data: Dict = None) -> Optional[int]:
        """Append entry to log."""
        with self.lock:
            if self.state != NodeState.LEADER:
                return None
            
            entry = LogEntry(
                index=len(self.log),
                term=self.current_term,
                command=command,
                data=data or {}
            )
            
            self.log.append(entry)
            return entry.index
    
    def receive_heartbeat(self, leader_id: str, leader_term: int) -> bool:
        """Receive leader heartbeat."""
        with self.lock:
            if leader_term > self.current_term:
                self.current_term = leader_term
                self.state = NodeState.FOLLOWER
                self.voted_for = None
            
            if leader_term >= self.current_term:
                self.election_timeout = time.time() + 1.5
                return True
            
            return False
    
    def request_vote(self, candidate_id: str, term: int) -> bool:
        """Handle vote request."""
        with self.lock:
            if term <= self.current_term:
                return False
            
            if self.voted_for is None or self.voted_for == candidate_id:
                self.voted_for = candidate_id
                self.current_term = term
                return True
            
            return False
    
    def start_election(self) -> bool:
        """Start leader election."""
        with self.lock:
            if time.time() < self.election_timeout:
                return False
            
            self.state = NodeState.CANDIDATE
            self.current_term += 1
            self.voted_for = self.node_id
            
            return True
    
    def become_leader(self) -> None:
        """Transition to leader."""
        with self.lock:
            self.state = NodeState.LEADER
            
            # Initialize leader state
            for peer in self.peer_ids:
                self.next_index[peer] = len(self.log)
                self.match_index[peer] = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get node state."""
        with self.lock:
            return {
                'node_id': self.node_id,
                'state': self.state.value,
                'term': self.current_term,
                'log_length': len(self.log),
                'commit_index': self.commit_index
            }

class ClusterCoordinator:
    """Coordinate distributed cluster."""
    
    def __init__(self, nodes: Dict[str, RaftNode]):
        self.nodes = nodes
        self.leader: Optional[str] = None
        self.lock = threading.RLock()
    
    def elect_leader(self) -> Optional[str]:
        """Elect cluster leader."""
        with self.lock:
            candidates = []
            
            for node_id, node in self.nodes.items():
                if node.start_election():
                    candidates.append(node_id)
            
            if not candidates:
                return self.leader
            
            # Simple majority voting
            votes: Dict[str, int] = defaultdict(int)
            
            for node_id in candidates:
                votes[node_id] += 1
            
            for voter_id, node in self.nodes.items():
                if voter_id not in candidates:
                    candidate = max(candidates, key=lambda x: self.nodes[x].current_term)
                    if node.request_vote(candidate, self.nodes[candidate].current_term):
                        votes[candidate] += 1
            
            # Check for majority
            majority = len(self.nodes) // 2 + 1
            
            for candidate_id, vote_count in votes.items():
                if vote_count >= majority:
                    self.leader = candidate_id
                    self.nodes[candidate_id].become_leader()
                    return candidate_id
            
            return None
    
    def send_heartbeats(self) -> None:
        """Send heartbeats from leader."""
        with self.lock:
            if not self.leader:
                return
            
            leader_node = self.nodes[self.leader]
            
            for peer_id in leader_node.peer_ids:
                self.nodes[peer_id].receive_heartbeat(
                    self.leader,
                    leader_node.current_term
                )
    
    def replicate_entry(self, command: str, data: Dict = None) -> Optional[int]:
        """Replicate entry across cluster."""
        with self.lock:
            if not self.leader:
                return None
            
            leader_node = self.nodes[self.leader]
            return leader_node.append_entry(command, data)
    
    def get_cluster_state(self) -> Dict[str, Any]:
        """Get cluster state."""
        with self.lock:
            return {
                'leader': self.leader,
                'nodes': {
                    node_id: node.get_state()
                    for node_id, node in self.nodes.items()
                }
            }

class DistributedLock:
    """Distributed lock implementation."""
    
    def __init__(self, lock_id: str, coordinator: ClusterCoordinator, ttl: float = 30.0):
        self.lock_id = lock_id
        self.coordinator = coordinator
        self.ttl = ttl
        self.holder: Optional[str] = None
        self.acquired_at: Optional[float] = None
        self.lock = threading.RLock()
    
    def acquire(self, requester_id: str) -> bool:
        """Acquire distributed lock."""
        with self.lock:
            # Check if lock is expired
            if self.holder and self.acquired_at:
                if time.time() - self.acquired_at > self.ttl:
                    self.holder = None
            
            if self.holder is None:
                self.holder = requester_id
                self.acquired_at = time.time()
                
                # Replicate lock state
                self.coordinator.replicate_entry(
                    "lock_acquire",
                    {'lock_id': self.lock_id, 'holder': requester_id}
                )
                
                return True
            
            return self.holder == requester_id
    
    def release(self, requester_id: str) -> bool:
        """Release distributed lock."""
        with self.lock:
            if self.holder != requester_id:
                return False
            
            self.holder = None
            self.acquired_at = None
            
            # Replicate lock state
            self.coordinator.replicate_entry(
                "lock_release",
                {'lock_id': self.lock_id}
            )
            
            return True
    
    def is_held(self) -> bool:
        """Check if lock is held."""
        with self.lock:
            if not self.holder:
                return False
            
            if time.time() - self.acquired_at > self.ttl:
                self.holder = None
                return False
            
            return True

class DistributedBarrier:
    """Barrier for distributed synchronization."""
    
    def __init__(self, barrier_id: str, num_parties: int,
                 coordinator: ClusterCoordinator):
        self.barrier_id = barrier_id
        self.num_parties = num_parties
        self.coordinator = coordinator
        self.arrived: List[str] = []
        self.lock = threading.RLock()
    
    def wait(self, party_id: str, timeout: float = 10.0) -> bool:
        """Wait at barrier."""
        start_time = time.time()
        
        with self.lock:
            if party_id not in self.arrived:
                self.arrived.append(party_id)
                
                # Replicate arrival
                self.coordinator.replicate_entry(
                    "barrier_arrive",
                    {'barrier_id': self.barrier_id, 'party_id': party_id}
                )
        
        # Wait for all parties
        while True:
            with self.lock:
                if len(self.arrived) >= self.num_parties:
                    self.arrived.clear()
                    return True
            
            if time.time() - start_time > timeout:
                return False
            
            time.sleep(0.1)

class QuorumManager:
    """Manage quorum-based decisions."""
    
    def __init__(self, cluster_size: int):
        self.cluster_size = cluster_size
        self.quorum_size = cluster_size // 2 + 1
        self.votes: Dict[str, Dict] = {}
        self.lock = threading.RLock()
    
    def start_vote(self, vote_id: str, question: str) -> None:
        """Start new vote."""
        with self.lock:
            self.votes[vote_id] = {
                'question': question,
                'responses': {},
                'started_at': time.time()
            }
    
    def cast_vote(self, vote_id: str, voter_id: str, response: bool) -> None:
        """Cast vote."""
        with self.lock:
            if vote_id in self.votes:
                self.votes[vote_id]['responses'][voter_id] = response
    
    def get_result(self, vote_id: str) -> Optional[bool]:
        """Get vote result."""
        with self.lock:
            if vote_id not in self.votes:
                return None
            
            vote = self.votes[vote_id]
            responses = list(vote['responses'].values())
            
            if len(responses) >= self.quorum_size:
                yes_votes = sum(1 for r in responses if r)
                return yes_votes > len(responses) // 2
            
            return None

# Example usage
if __name__ == "__main__":
    import json
    
    # Create cluster
    node_ids = ['node1', 'node2', 'node3']
    nodes = {node_id: RaftNode(node_id, node_ids) for node_id in node_ids}
    
    coordinator = ClusterCoordinator(nodes)
    
    # Elect leader
    leader = coordinator.elect_leader()
    print(f"Leader elected: {leader}")
    
    # Get cluster state
    state = coordinator.get_cluster_state()
    print(f"\nCluster State:")
    print(json.dumps(state, indent=2, default=str))
    
    # Test distributed lock
    lock = DistributedLock("resource1", coordinator)
    acquired = lock.acquire("client1")
    print(f"\nLock acquired: {acquired}")
    
    # Test quorum voting
    quorum = QuorumManager(3)
    quorum.start_vote("vote1", "Should we scale up?")
    quorum.cast_vote("vote1", "node1", True)
    quorum.cast_vote("vote1", "node2", True)
    result = quorum.get_result("vote1")
    print(f"Vote result: {result}")
