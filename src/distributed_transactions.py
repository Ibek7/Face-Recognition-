# Distributed Transactions

import threading
import time
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

class TransactionState(Enum):
    """Transaction states."""
    INITIAL = "initial"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"

@dataclass
class TransactionLog:
    """Transaction log entry."""
    tx_id: str
    operation: str
    resource_id: str
    data: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    redo: bool = False

@dataclass
class ParticipantVote:
    """Vote from transaction participant."""
    participant_id: str
    vote: str  # 'yes' or 'no'
    timestamp: float = field(default_factory=time.time)

class Participant:
    """Transaction participant."""
    
    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.resources: Dict[str, Dict] = {}
        self.locks: Dict[str, str] = {}  # resource_id -> tx_id
        self.prepared_resources: Dict[str, Dict] = {}
        self.lock = threading.RLock()
    
    def prepare(self, tx_id: str, operations: List[Dict]) -> Tuple[bool, str]:
        """Prepare for transaction."""
        with self.lock:
            try:
                # Validate operations
                for op in operations:
                    resource_id = op.get('resource_id')
                    
                    # Check if resource is locked
                    if resource_id in self.locks:
                        if self.locks[resource_id] != tx_id:
                            return False, "Resource locked"
                    
                    # Lock resource
                    self.locks[resource_id] = tx_id
                
                # Save prepared state
                self.prepared_resources[tx_id] = {
                    'operations': operations,
                    'timestamp': time.time()
                }
                
                return True, "Ready to commit"
            
            except Exception as e:
                return False, str(e)
    
    def commit(self, tx_id: str) -> bool:
        """Commit transaction."""
        with self.lock:
            if tx_id not in self.prepared_resources:
                return False
            
            try:
                prepared = self.prepared_resources[tx_id]
                
                # Apply operations
                for op in prepared['operations']:
                    resource_id = op.get('resource_id')
                    data = op.get('data', {})
                    
                    if resource_id not in self.resources:
                        self.resources[resource_id] = {}
                    
                    self.resources[resource_id].update(data)
                
                # Release locks
                for op in prepared['operations']:
                    resource_id = op.get('resource_id')
                    if resource_id in self.locks:
                        del self.locks[resource_id]
                
                # Clean up prepared state
                del self.prepared_resources[tx_id]
                
                return True
            
            except Exception:
                return False
    
    def abort(self, tx_id: str) -> None:
        """Abort transaction."""
        with self.lock:
            # Release locks
            for resource_id in list(self.locks.keys()):
                if self.locks[resource_id] == tx_id:
                    del self.locks[resource_id]
            
            # Clean up prepared state
            if tx_id in self.prepared_resources:
                del self.prepared_resources[tx_id]

class TwoPhaseCommitCoordinator:
    """Two-phase commit coordinator."""
    
    def __init__(self, coordinator_id: str = "coordinator"):
        self.coordinator_id = coordinator_id
        self.participants: Dict[str, Participant] = {}
        self.transactions: Dict[str, Dict] = {}
        self.transaction_log: List[TransactionLog] = []
        self.lock = threading.RLock()
    
    def register_participant(self, participant: Participant) -> None:
        """Register participant."""
        with self.lock:
            self.participants[participant.participant_id] = participant
    
    def begin_transaction(self) -> str:
        """Begin new transaction."""
        tx_id = str(uuid.uuid4())
        
        with self.lock:
            self.transactions[tx_id] = {
                'state': TransactionState.INITIAL,
                'created_at': time.time(),
                'votes': {}
            }
        
        return tx_id
    
    def execute_transaction(self, tx_id: str, operations: List[Dict]) -> bool:
        """Execute two-phase commit."""
        if tx_id not in self.transactions:
            return False
        
        # Phase 1: Prepare
        if not self._phase1_prepare(tx_id, operations):
            self._abort(tx_id)
            return False
        
        # Phase 2: Commit
        if not self._phase2_commit(tx_id):
            self._abort(tx_id)
            return False
        
        with self.lock:
            self.transactions[tx_id]['state'] = TransactionState.COMMITTED
        
        return True
    
    def _phase1_prepare(self, tx_id: str, operations: List[Dict]) -> bool:
        """Phase 1: Prepare."""
        with self.lock:
            self.transactions[tx_id]['state'] = TransactionState.PREPARING
        
        votes = {}
        
        with self.lock:
            for participant_id, participant in self.participants.items():
                ready, message = participant.prepare(tx_id, operations)
                vote = "yes" if ready else "no"
                votes[participant_id] = vote
                
                self.transactions[tx_id]['votes'] = votes
        
        # Check if all participants voted yes
        return all(v == "yes" for v in votes.values())
    
    def _phase2_commit(self, tx_id: str) -> bool:
        """Phase 2: Commit."""
        with self.lock:
            self.transactions[tx_id]['state'] = TransactionState.COMMITTING
        
        success = True
        
        with self.lock:
            for participant in self.participants.values():
                if not participant.commit(tx_id):
                    success = False
        
        return success
    
    def _abort(self, tx_id: str) -> None:
        """Abort transaction."""
        with self.lock:
            self.transactions[tx_id]['state'] = TransactionState.ABORTING
        
        with self.lock:
            for participant in self.participants.values():
                participant.abort(tx_id)
        
        with self.lock:
            self.transactions[tx_id]['state'] = TransactionState.ABORTED
    
    def get_transaction_status(self, tx_id: str) -> Dict:
        """Get transaction status."""
        with self.lock:
            if tx_id not in self.transactions:
                return {}
            
            return self.transactions[tx_id]

class TransactionRecovery:
    """Transaction recovery and crash handling."""
    
    def __init__(self):
        self.transaction_log: List[TransactionLog] = []
        self.checkpoint_interval = 100
        self.lock = threading.RLock()
    
    def log_operation(self, tx_id: str, operation: str, resource_id: str,
                     data: Dict = None) -> None:
        """Log transaction operation."""
        data = data or {}
        
        with self.lock:
            entry = TransactionLog(tx_id, operation, resource_id, data)
            self.transaction_log.append(entry)
    
    def create_checkpoint(self) -> Dict:
        """Create recovery checkpoint."""
        with self.lock:
            if len(self.transaction_log) > self.checkpoint_interval:
                # Simulated checkpoint
                checkpoint = {
                    'timestamp': time.time(),
                    'log_entries': len(self.transaction_log),
                    'entries': self.transaction_log.copy()
                }
                
                # Clear old entries
                self.transaction_log = []
                
                return checkpoint
        
        return {}
    
    def recover_from_checkpoint(self, checkpoint: Dict) -> List[TransactionLog]:
        """Recover from checkpoint."""
        with self.lock:
            return checkpoint.get('entries', [])

class DistributedLock:
    """Distributed lock for transaction isolation."""
    
    def __init__(self, lock_id: str):
        self.lock_id = lock_id
        self.owner: Optional[str] = None
        self.acquired_at = 0
        self.lock = threading.RLock()
    
    def acquire(self, tx_id: str, timeout: float = 10.0) -> bool:
        """Acquire lock."""
        start_time = time.time()
        
        while True:
            with self.lock:
                if self.owner is None:
                    self.owner = tx_id
                    self.acquired_at = time.time()
                    return True
                
                if self.owner == tx_id:
                    return True
            
            if time.time() - start_time > timeout:
                return False
            
            time.sleep(0.01)
    
    def release(self, tx_id: str) -> bool:
        """Release lock."""
        with self.lock:
            if self.owner == tx_id:
                self.owner = None
                return True
            
            return False

# Example usage
if __name__ == "__main__":
    # Create coordinator and participants
    coordinator = TwoPhaseCommitCoordinator()
    
    p1 = Participant("participant-1")
    p2 = Participant("participant-2")
    
    coordinator.register_participant(p1)
    coordinator.register_participant(p2)
    
    # Begin transaction
    tx_id = coordinator.begin_transaction()
    print(f"Transaction ID: {tx_id}")
    
    # Define operations
    operations = [
        {'resource_id': 'resource-1', 'data': {'value': 100}},
        {'resource_id': 'resource-2', 'data': {'value': 200}}
    ]
    
    # Execute transaction
    success = coordinator.execute_transaction(tx_id, operations)
    print(f"Transaction success: {success}")
    
    # Get status
    status = coordinator.get_transaction_status(tx_id)
    print(f"Transaction status: {json.dumps(status, indent=2, default=str)}")
