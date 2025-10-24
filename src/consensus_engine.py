# Distributed Consensus Algorithms

import threading
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json

class NodeState(Enum):
    """Raft node states."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

class LogEntryType(Enum):
    """Log entry types."""
    COMMAND = "command"
    CONFIG = "config"

@dataclass
class LogEntry:
    """Raft log entry."""
    term: int
    index: int
    entry_type: LogEntryType
    data: Dict
    committed: bool = False

@dataclass
class RaftState:
    """Raft state."""
    current_term: int = 0
    voted_for: Optional[str] = None
    log_entries: List[LogEntry] = field(default_factory=list)
    commit_index: int = 0
    last_applied: int = 0

class RaftNode:
    """Raft consensus node."""
    
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.state = RaftState()
        self.node_state = NodeState.FOLLOWER
        self.leader_id: Optional[str] = None
        self.last_heartbeat = time.time()
        self.election_timeout = 1.5 + hash(node_id) % 10 * 0.1
        self.lock = threading.RLock()
        self.state_machine: Dict = {}
    
    def request_vote(self, term: int, candidate_id: str,
                    last_log_index: int, last_log_term: int) -> Tuple[int, bool]:
        """Handle vote request."""
        with self.lock:
            if term > self.state.current_term:
                self.state.current_term = term
                self.state.voted_for = None
            
            if term < self.state.current_term:
                return self.state.current_term, False
            
            if self.state.voted_for is None or self.state.voted_for == candidate_id:
                if self._is_log_up_to_date(last_log_index, last_log_term):
                    self.state.voted_for = candidate_id
                    return term, True
            
            return self.state.current_term, False
    
    def append_entries(self, term: int, leader_id: str,
                      prev_log_index: int, prev_log_term: int,
                      entries: List[LogEntry], leader_commit: int) -> Tuple[int, bool]:
        """Handle append entries RPC."""
        with self.lock:
            if term > self.state.current_term:
                self.state.current_term = term
                self.state.voted_for = None
            
            if term < self.state.current_term:
                return self.state.current_term, False
            
            self.node_state = NodeState.FOLLOWER
            self.leader_id = leader_id
            self.last_heartbeat = time.time()
            
            # Check log consistency
            if prev_log_index > 0:
                if prev_log_index > len(self.state.log_entries) - 1:
                    return term, False
                
                if self.state.log_entries[prev_log_index].term != prev_log_term:
                    # Delete conflicting entries
                    self.state.log_entries = self.state.log_entries[:prev_log_index]
                    return term, False
            
            # Append new entries
            for entry in entries:
                if entry.index > len(self.state.log_entries) - 1:
                    self.state.log_entries.append(entry)
            
            # Update commit index
            if leader_commit > self.state.commit_index:
                self.state.commit_index = min(leader_commit, len(self.state.log_entries) - 1)
            
            return term, True
    
    def _is_log_up_to_date(self, last_log_index: int, last_log_term: int) -> bool:
        """Check if candidate log is up to date."""
        if not self.state.log_entries:
            return True
        
        last_entry = self.state.log_entries[-1]
        if last_log_term > last_entry.term:
            return True
        
        return last_log_term == last_entry.term and last_log_index >= last_entry.index
    
    def become_leader(self) -> None:
        """Transition to leader."""
        with self.lock:
            self.node_state = NodeState.LEADER
            self.leader_id = self.node_id
            self.state.voted_for = None
    
    def become_candidate(self) -> None:
        """Transition to candidate."""
        with self.lock:
            self.node_state = NodeState.CANDIDATE
            self.state.current_term += 1
            self.state.voted_for = self.node_id
            self.last_heartbeat = time.time()
    
    def replicate_log(self, entry: LogEntry) -> None:
        """Replicate log entry to followers."""
        with self.lock:
            if self.node_state != NodeState.LEADER:
                raise RuntimeError("Only leader can replicate")
            
            self.state.log_entries.append(entry)
    
    def apply_entries(self) -> None:
        """Apply committed entries to state machine."""
        with self.lock:
            while self.state.last_applied < self.state.commit_index:
                entry = self.state.log_entries[self.state.last_applied]
                self.state_machine.update(entry.data)
                self.state.last_applied += 1
    
    def get_state(self) -> Dict:
        """Get node state."""
        with self.lock:
            return {
                'node_id': self.node_id,
                'state': self.node_state.value,
                'term': self.state.current_term,
                'log_length': len(self.state.log_entries),
                'commit_index': self.state.commit_index,
                'leader': self.leader_id
            }

class PaxosNode:
    """Paxos consensus node."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.role = "acceptor"  # proposer, acceptor, learner
        self.prepared_ballot = 0
        self.accepted_ballot = 0
        self.accepted_value = None
        self.lock = threading.RLock()
    
    def prepare(self, ballot_num: int) -> Tuple[int, int, Optional[Dict]]:
        """Prepare phase."""
        with self.lock:
            if ballot_num > self.prepared_ballot:
                self.prepared_ballot = ballot_num
                return ballot_num, self.accepted_ballot, self.accepted_value
            
            return self.prepared_ballot, self.accepted_ballot, self.accepted_value
    
    def accept(self, ballot_num: int, value: Dict) -> bool:
        """Accept phase."""
        with self.lock:
            if ballot_num >= self.prepared_ballot:
                self.accepted_ballot = ballot_num
                self.accepted_value = value
                return True
            
            return False

class ConsensusCluster:
    """Consensus cluster coordinator."""
    
    def __init__(self, node_ids: List[str], algorithm: str = "raft"):
        self.node_ids = node_ids
        self.algorithm = algorithm
        self.nodes: Dict[str, RaftNode] = {}
        self.quorum_size = len(node_ids) // 2 + 1
        self.lock = threading.RLock()
        
        # Initialize nodes
        for node_id in node_ids:
            if algorithm == "raft":
                self.nodes[node_id] = RaftNode(node_id, node_ids)
    
    def elect_leader(self) -> Optional[str]:
        """Elect leader through consensus."""
        votes: Dict[str, int] = {}
        
        with self.lock:
            for node_id in self.node_ids:
                node = self.nodes[node_id]
                
                # Check if election timeout expired
                if time.time() - node.last_heartbeat > node.election_timeout:
                    node.become_candidate()
                    votes[node_id] = 1
                
                # Simulate vote collection
                for peer_id in node.peers:
                    if peer_id != node_id and peer_id in self.nodes:
                        peer = self.nodes[peer_id]
                        term, granted = peer.request_vote(
                            node.state.current_term,
                            node_id,
                            len(node.state.log_entries) - 1,
                            node.state.log_entries[-1].term if node.state.log_entries else 0
                        )
                        
                        if granted:
                            votes[node_id] = votes.get(node_id, 0) + 1
            
            # Check if any node has quorum
            for node_id, vote_count in votes.items():
                if vote_count >= self.quorum_size:
                    self.nodes[node_id].become_leader()
                    return node_id
        
        return None
    
    def replicate_entry(self, key: str, value: str) -> bool:
        """Replicate entry to all nodes."""
        with self.lock:
            # Find leader
            leader = None
            for node in self.nodes.values():
                if node.node_state == NodeState.LEADER:
                    leader = node
                    break
            
            if not leader:
                return False
            
            # Create log entry
            entry = LogEntry(
                term=leader.state.current_term,
                index=len(leader.state.log_entries),
                entry_type=LogEntryType.COMMAND,
                data={key: value}
            )
            
            # Replicate to followers
            leader.replicate_log(entry)
            replicas = 1
            
            for node in self.nodes.values():
                if node.node_id != leader.node_id:
                    # Simulate replication
                    prev_index = len(leader.state.log_entries) - 2
                    prev_term = leader.state.log_entries[prev_index].term if prev_index >= 0 else 0
                    
                    term, success = node.append_entries(
                        leader.state.current_term,
                        leader.node_id,
                        prev_index,
                        prev_term,
                        [entry],
                        leader.state.commit_index
                    )
                    
                    if success:
                        replicas += 1
            
            return replicas >= self.quorum_size
    
    def get_cluster_status(self) -> Dict:
        """Get cluster status."""
        with self.lock:
            status = {}
            for node_id, node in self.nodes.items():
                status[node_id] = node.get_state()
            
            return status

# Example usage
if __name__ == "__main__":
    # Create cluster
    cluster = ConsensusCluster(["node-1", "node-2", "node-3"], "raft")
    
    # Simulate election
    leader = cluster.elect_leader()
    print(f"Elected leader: {leader}")
    
    # Get cluster status
    status = cluster.get_cluster_status()
    print(f"\nCluster Status:")
    for node_id, state in status.items():
        print(f"  {node_id}: {json.dumps(state, indent=2)}")
    
    # Replicate entry
    success = cluster.replicate_entry("key-1", "value-1")
    print(f"\nReplication success: {success}")
    
    # Final status
    final_status = cluster.get_cluster_status()
    print(f"\nFinal Cluster Status:")
    for node_id, state in final_status.items():
        print(f"  {node_id}: {json.dumps(state)}")
