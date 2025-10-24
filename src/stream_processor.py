# Stream Processing Engine

import threading
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json

class WindowType(Enum):
    """Stream window types."""
    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"

@dataclass
class StreamEvent:
    """Stream event."""
    event_id: str
    timestamp: float
    data: Dict
    source: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'data': self.data,
            'source': self.source
        }

class StreamWindow:
    """Stream window for aggregation."""
    
    def __init__(self, window_id: str, start_time: float, end_time: float):
        self.window_id = window_id
        self.start_time = start_time
        self.end_time = end_time
        self.events: List[StreamEvent] = []
        self.result: Optional[Dict] = None
    
    def add_event(self, event: StreamEvent) -> None:
        """Add event to window."""
        if self.start_time <= event.timestamp <= self.end_time:
            self.events.append(event)
    
    def is_closed(self, current_time: float) -> bool:
        """Check if window is closed."""
        return current_time > self.end_time
    
    def aggregate(self, aggregation_func: Callable) -> Dict:
        """Aggregate window events."""
        self.result = aggregation_func(self.events)
        return self.result

class TumblingWindow:
    """Tumbling (fixed-size) window."""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.windows: Dict[int, StreamWindow] = {}
        self.lock = threading.RLock()
    
    def add_event(self, event: StreamEvent) -> None:
        """Add event to appropriate window."""
        window_id = int(event.timestamp // self.window_size)
        
        with self.lock:
            if window_id not in self.windows:
                start_time = window_id * self.window_size
                end_time = start_time + self.window_size
                self.windows[window_id] = StreamWindow(
                    f"window-{window_id}",
                    start_time,
                    end_time
                )
            
            self.windows[window_id].add_event(event)
    
    def get_closed_windows(self, current_time: float) -> List[StreamWindow]:
        """Get closed windows."""
        with self.lock:
            closed = [w for w in self.windows.values()
                     if w.is_closed(current_time)]
            return closed

class SlidingWindow:
    """Sliding window."""
    
    def __init__(self, window_size: int, slide_size: int):
        self.window_size = window_size
        self.slide_size = slide_size
        self.events: deque = deque()
        self.windows: Dict[int, StreamWindow] = {}
        self.lock = threading.RLock()
    
    def add_event(self, event: StreamEvent) -> None:
        """Add event to sliding window."""
        with self.lock:
            self.events.append(event)
            
            # Remove old events
            cutoff_time = event.timestamp - self.window_size
            while self.events and self.events[0].timestamp < cutoff_time:
                self.events.popleft()
    
    def get_current_window(self, current_time: float) -> List[StreamEvent]:
        """Get current window events."""
        with self.lock:
            return [e for e in self.events
                   if current_time - self.window_size <= e.timestamp <= current_time]

class SessionWindow:
    """Session window (event-triggered)."""
    
    def __init__(self, session_timeout: int):
        self.session_timeout = session_timeout
        self.sessions: Dict[str, List[StreamEvent]] = {}
        self.last_event_time: Dict[str, float] = {}
        self.lock = threading.RLock()
    
    def add_event(self, event: StreamEvent) -> None:
        """Add event to session."""
        session_id = event.data.get('session_id', 'default')
        
        with self.lock:
            current_time = event.timestamp
            
            # Check if session is still active
            if session_id in self.last_event_time:
                time_diff = current_time - self.last_event_time[session_id]
                if time_diff > self.session_timeout:
                    # Start new session
                    self.sessions[session_id] = []
            else:
                self.sessions[session_id] = []
            
            self.sessions[session_id].append(event)
            self.last_event_time[session_id] = current_time
    
    def get_sessions(self) -> Dict[str, List[StreamEvent]]:
        """Get all sessions."""
        with self.lock:
            return self.sessions.copy()

class StreamProcessor:
    """Stream processor."""
    
    def __init__(self):
        self.sources: Dict[str, Callable] = {}
        self.operators: List[Callable] = []
        self.sinks: List[Callable] = []
        self.event_queue: deque = deque()
        self.lock = threading.RLock()
        self.running = False
    
    def add_source(self, source_name: str, source_func: Callable) -> None:
        """Add event source."""
        with self.lock:
            self.sources[source_name] = source_func
    
    def add_operator(self, operator: Callable) -> None:
        """Add stream operator (map, filter, etc)."""
        self.operators.append(operator)
    
    def add_sink(self, sink_func: Callable) -> None:
        """Add output sink."""
        self.sinks.append(sink_func)
    
    def process_event(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Process event through pipeline."""
        current = event
        
        # Apply operators
        for operator in self.operators:
            if current is None:
                break
            current = operator(current)
        
        # Send to sinks
        if current is not None:
            for sink in self.sinks:
                sink(current)
        
        return current
    
    def start(self) -> None:
        """Start processing."""
        with self.lock:
            self.running = True
        
        # Generate events from sources
        for source_name, source_func in self.sources.items():
            event = source_func()
            if event:
                self.process_event(event)

class StreamAggregation:
    """Stream aggregation operations."""
    
    @staticmethod
    def sum_aggregation(events: List[StreamEvent]) -> Dict:
        """Sum aggregation."""
        total = sum(e.data.get('value', 0) for e in events)
        return {
            'aggregation': 'sum',
            'value': total,
            'count': len(events)
        }
    
    @staticmethod
    def average_aggregation(events: List[StreamEvent]) -> Dict:
        """Average aggregation."""
        if not events:
            return {'aggregation': 'average', 'value': 0, 'count': 0}
        
        values = [e.data.get('value', 0) for e in events]
        avg = sum(values) / len(values)
        
        return {
            'aggregation': 'average',
            'value': avg,
            'count': len(events)
        }
    
    @staticmethod
    def count_aggregation(events: List[StreamEvent]) -> Dict:
        """Count aggregation."""
        return {
            'aggregation': 'count',
            'value': len(events)
        }
    
    @staticmethod
    def max_aggregation(events: List[StreamEvent]) -> Dict:
        """Max aggregation."""
        values = [e.data.get('value', 0) for e in events]
        max_val = max(values) if values else 0
        
        return {
            'aggregation': 'max',
            'value': max_val
        }

class StreamJoin:
    """Stream join operation."""
    
    def __init__(self, join_window: int = 5):
        self.join_window = join_window
        self.left_stream: List[StreamEvent] = []
        self.right_stream: List[StreamEvent] = []
        self.lock = threading.RLock()
    
    def add_left_event(self, event: StreamEvent) -> None:
        """Add event to left stream."""
        with self.lock:
            self.left_stream.append(event)
            # Clean old events
            cutoff_time = event.timestamp - self.join_window
            self.left_stream = [e for e in self.left_stream
                               if e.timestamp >= cutoff_time]
    
    def add_right_event(self, event: StreamEvent) -> None:
        """Add event to right stream."""
        with self.lock:
            self.right_stream.append(event)
            # Clean old events
            cutoff_time = event.timestamp - self.join_window
            self.right_stream = [e for e in self.right_stream
                                if e.timestamp >= cutoff_time]
    
    def get_joined_events(self, join_key: str) -> List[Dict]:
        """Get joined events."""
        with self.lock:
            joined = []
            
            for left_event in self.left_stream:
                left_key = left_event.data.get(join_key)
                
                for right_event in self.right_stream:
                    right_key = right_event.data.get(join_key)
                    
                    if left_key == right_key:
                        joined.append({
                            'left': left_event.to_dict(),
                            'right': right_event.to_dict()
                        })
            
            return joined

# Example usage
if __name__ == "__main__":
    # Create processor
    processor = StreamProcessor()
    
    # Define operators
    def filter_op(event: StreamEvent) -> Optional[StreamEvent]:
        """Filter events with value > 50."""
        if event.data.get('value', 0) > 50:
            return event
        return None
    
    def map_op(event: StreamEvent) -> StreamEvent:
        """Map operation."""
        event.data['processed'] = True
        return event
    
    processor.add_operator(filter_op)
    processor.add_operator(map_op)
    
    # Define sink
    results = []
    def sink_op(event: StreamEvent) -> None:
        """Output sink."""
        results.append(event.to_dict())
    
    processor.add_sink(sink_op)
    
    # Create tumbling window
    window = TumblingWindow(window_size=10)
    
    # Add events
    for i in range(5):
        event = StreamEvent(
            event_id=f"event-{i}",
            timestamp=time.time() + i,
            data={'value': 50 + i * 10}
        )
        window.add_event(event)
        processor.process_event(event)
    
    print(f"Processed events: {len(results)}")
    for result in results:
        print(f"  {json.dumps(result, indent=2)}")
