# State Machine & FSM Framework

import threading
import time
import json
from typing import Dict, Any, Optional, Callable, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

@dataclass
class StateTransition:
    """Definition of state transition."""
    from_state: str
    to_state: str
    event: str
    condition: Optional[Callable] = None
    action: Optional[Callable] = None
    guard: Optional[Callable] = None
    
    def can_transition(self, context: Dict = None) -> bool:
        """Check if transition is allowed."""
        if self.guard:
            return self.guard(context or {})
        return True
    
    def execute_action(self, context: Dict = None) -> None:
        """Execute transition action."""
        if self.action:
            self.action(context or {})

@dataclass
class StateTransitionEvent:
    """Event record of state transition."""
    timestamp: float = field(default_factory=time.time)
    from_state: str = ""
    to_state: str = ""
    event: str = ""
    duration_sec: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'from_state': self.from_state,
            'to_state': self.to_state,
            'event': self.event,
            'duration_sec': self.duration_sec
        }

class State:
    """State in FSM."""
    
    def __init__(self, name: str, is_initial: bool = False, is_final: bool = False):
        self.name = name
        self.is_initial = is_initial
        self.is_final = is_final
        self.entry_action: Optional[Callable] = None
        self.exit_action: Optional[Callable] = None
        self.timeout_sec: Optional[float] = None
    
    def on_entry(self, context: Dict = None) -> None:
        """Execute entry action."""
        if self.entry_action:
            self.entry_action(context or {})
    
    def on_exit(self, context: Dict = None) -> None:
        """Execute exit action."""
        if self.exit_action:
            self.exit_action(context or {})

class StateMachine:
    """Finite State Machine implementation."""
    
    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self.states: Dict[str, State] = {}
        self.transitions: List[StateTransition] = []
        self.current_state: Optional[str] = None
        self.context: Dict[str, Any] = {}
        self.history: List[StateTransitionEvent] = []
        
        self.state_enter_time: Optional[float] = None
        self.lock = threading.RLock()
    
    def add_state(self, state: State) -> None:
        """Add state to machine."""
        with self.lock:
            self.states[state.name] = state
            
            if state.is_initial:
                self.current_state = state.name
    
    def add_transition(self, transition: StateTransition) -> None:
        """Add transition to machine."""
        with self.lock:
            self.transitions.append(transition)
    
    def trigger_event(self, event: str, context: Dict = None) -> bool:
        """Trigger event to cause state transition."""
        with self.lock:
            if not self.current_state:
                return False
            
            # Find matching transition
            transition = self._find_transition(self.current_state, event)
            
            if not transition:
                return False
            
            # Check guard condition
            if not transition.can_transition(context):
                return False
            
            # Execute transition
            old_state = self.current_state
            
            # Exit old state
            state = self.states[old_state]
            state.on_exit(context or self.context)
            
            # Execute action
            transition.execute_action(context or self.context)
            
            # Enter new state
            new_state = transition.to_state
            self.current_state = new_state
            self.state_enter_time = time.time()
            
            state = self.states[new_state]
            state.on_entry(context or self.context)
            
            # Record in history
            event_record = StateTransitionEvent(
                from_state=old_state,
                to_state=new_state,
                event=event,
                context=context or {}
            )
            self.history.append(event_record)
            
            return True
    
    def _find_transition(self, from_state: str, event: str) -> Optional[StateTransition]:
        """Find transition for given event from state."""
        for transition in self.transitions:
            if transition.from_state == from_state and transition.event == event:
                return transition
        
        return None
    
    def get_current_state(self) -> str:
        """Get current state."""
        with self.lock:
            return self.current_state or ""
    
    def get_valid_events(self) -> List[str]:
        """Get valid events from current state."""
        with self.lock:
            if not self.current_state:
                return []
            
            events = []
            for transition in self.transitions:
                if transition.from_state == self.current_state:
                    events.append(transition.event)
            
            return events
    
    def is_final_state(self) -> bool:
        """Check if current state is final."""
        with self.lock:
            if not self.current_state:
                return False
            
            state = self.states.get(self.current_state)
            return state.is_final if state else False
    
    def get_status(self) -> Dict:
        """Get FSM status."""
        with self.lock:
            time_in_state = None
            if self.state_enter_time:
                time_in_state = time.time() - self.state_enter_time
            
            return {
                'machine_id': self.machine_id,
                'current_state': self.current_state,
                'time_in_state_sec': time_in_state,
                'valid_events': self.get_valid_events(),
                'is_final': self.is_final_state(),
                'history_length': len(self.history)
            }
    
    def get_history(self) -> List[Dict]:
        """Get transition history."""
        with self.lock:
            return [event.to_dict() for event in self.history]

class StateMachineBuilder:
    """Build state machines fluently."""
    
    def __init__(self, machine_id: str):
        self.machine = StateMachine(machine_id)
    
    def state(self, name: str, initial: bool = False,
             final: bool = False) -> 'StateMachineBuilder':
        """Add state."""
        state = State(name, initial, final)
        self.machine.add_state(state)
        return self
    
    def on_entry(self, state_name: str, action: Callable) -> 'StateMachineBuilder':
        """Set entry action for state."""
        if state_name in self.machine.states:
            self.machine.states[state_name].entry_action = action
        return self
    
    def on_exit(self, state_name: str, action: Callable) -> 'StateMachineBuilder':
        """Set exit action for state."""
        if state_name in self.machine.states:
            self.machine.states[state_name].exit_action = action
        return self
    
    def transition(self, from_state: str, to_state: str, event: str,
                  action: Callable = None, guard: Callable = None) -> 'StateMachineBuilder':
        """Add transition."""
        trans = StateTransition(
            from_state=from_state,
            to_state=to_state,
            event=event,
            action=action,
            guard=guard
        )
        self.machine.add_transition(trans)
        return self
    
    def build(self) -> StateMachine:
        """Build and return FSM."""
        return self.machine

class HierarchicalStateMachine:
    """Hierarchical FSM with nested states."""
    
    def __init__(self, root_id: str):
        self.root_machine = StateMachine(root_id)
        self.submachines: Dict[str, StateMachine] = {}
        self.lock = threading.RLock()
    
    def add_submachine(self, parent_state: str, submachine: StateMachine) -> None:
        """Add submachine to parent state."""
        with self.lock:
            key = f"{parent_state}:{submachine.machine_id}"
            self.submachines[key] = submachine
    
    def trigger_event(self, event: str, context: Dict = None) -> bool:
        """Trigger event in hierarchical FSM."""
        current_state = self.root_machine.get_current_state()
        
        # Try submachine first
        key = f"{current_state}:*"
        submachine_key = None
        
        with self.lock:
            for k in self.submachines.keys():
                if k.startswith(current_state + ":"):
                    submachine_key = k
                    break
        
        if submachine_key:
            submachine = self.submachines[submachine_key]
            if submachine.trigger_event(event, context):
                return True
        
        # Try root machine
        return self.root_machine.trigger_event(event, context)

class StateMachineFactory:
    """Factory for creating common FSM patterns."""
    
    @staticmethod
    def create_order_fsm() -> StateMachine:
        """Create order processing FSM."""
        builder = StateMachineBuilder("order_fsm")
        
        machine = (builder
                  .state("pending", initial=True)
                  .state("processing")
                  .state("completed", final=True)
                  .state("cancelled", final=True)
                  .transition("pending", "processing", "process")
                  .transition("processing", "completed", "complete")
                  .transition("pending", "cancelled", "cancel")
                  .transition("processing", "cancelled", "cancel")
                  .build())
        
        return machine
    
    @staticmethod
    def create_traffic_light_fsm() -> StateMachine:
        """Create traffic light FSM."""
        builder = StateMachineBuilder("traffic_light")
        
        machine = (builder
                  .state("red", initial=True)
                  .state("yellow")
                  .state("green")
                  .transition("red", "green", "timer")
                  .transition("green", "yellow", "timer")
                  .transition("yellow", "red", "timer")
                  .build())
        
        return machine

# Example usage
if __name__ == "__main__":
    # Create FSM using builder
    def on_processing():
        print("Starting processing...")
    
    def on_complete():
        print("Order completed!")
    
    builder = StateMachineBuilder("order_process")
    fsm = (builder
           .state("pending", initial=True)
           .state("processing")
           .state("completed", final=True)
           .on_entry("processing", lambda ctx: print("Entering processing"))
           .transition("pending", "processing", "start", action=on_processing)
           .transition("processing", "completed", "finish", action=on_complete)
           .build())
    
    # Use FSM
    print(f"Initial state: {fsm.get_current_state()}")
    print(f"Valid events: {fsm.get_valid_events()}")
    
    # Trigger transitions
    fsm.trigger_event("start")
    print(f"After start: {fsm.get_current_state()}")
    
    fsm.trigger_event("finish")
    print(f"After finish: {fsm.get_current_state()}")
    print(f"Is final: {fsm.is_final_state()}")
    
    # Get history
    history = fsm.get_history()
    print(f"\nTransition History:")
    print(json.dumps(history, indent=2, default=str))
