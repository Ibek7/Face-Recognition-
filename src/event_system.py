# Event-Driven Architecture System

import threading
import time
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
import queue

class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class EventStatus(Enum):
    """Event status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Event:
    """Event in system."""
    event_type: str
    source: str
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: EventStatus = EventStatus.PENDING
    event_id: str = field(default_factory=lambda: str(id(object())))
    
    def __lt__(self, other):
        """Compare events by priority (for queue)."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.timestamp < other.timestamp
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'source': self.source,
            'timestamp': self.timestamp,
            'priority': self.priority.name,
            'status': self.status.value,
            'data': self.data,
            'metadata': self.metadata
        }

class EventHandler(ABC):
    """Base class for event handlers."""
    
    @abstractmethod
    def handle(self, event: Event) -> bool:
        """Handle event. Return True if handled successfully."""
        pass
    
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if handler can handle event."""
        pass

class EventBus:
    """Central event bus for pub/sub communication."""
    
    def __init__(self, max_workers: int = 4, queue_size: int = 1000):
        self.handlers: Dict[str, List[EventHandler]] = {}
        self.event_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_size)
        self.event_history: List[Event] = []
        self.max_workers = max_workers
        self.is_running = False
        self.worker_threads: List[threading.Thread] = []
        self.lock = threading.RLock()
        self.event_count = 0
        self.processed_count = 0
        self.failed_count = 0
    
    def subscribe(self, event_type: str, handler: EventHandler):
        """Subscribe handler to event type."""
        with self.lock:
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            
            if handler not in self.handlers[event_type]:
                self.handlers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: EventHandler):
        """Unsubscribe handler from event type."""
        with self.lock:
            if event_type in self.handlers:
                self.handlers[event_type] = [
                    h for h in self.handlers[event_type] if h != handler
                ]
    
    def publish(self, event: Event) -> str:
        """Publish event to bus."""
        with self.lock:
            self.event_count += 1
            event.metadata['published_at'] = time.time()
        
        try:
            self.event_queue.put((event.priority.value, event), block=False)
        except queue.Full:
            with self.lock:
                self.failed_count += 1
            raise Exception(f"Event queue full, could not publish event {event.event_id}")
        
        return event.event_id
    
    def start(self):
        """Start event processing."""
        if self.is_running:
            return
        
        self.is_running = True
        
        for _ in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.worker_threads.append(worker)
    
    def stop(self):
        """Stop event processing."""
        self.is_running = False
        
        for worker in self.worker_threads:
            worker.join(timeout=5)
        
        self.worker_threads.clear()
    
    def _worker_loop(self):
        """Worker loop for processing events."""
        while self.is_running:
            try:
                _, event = self.event_queue.get(timeout=1)
                self._process_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in worker loop: {e}")
    
    def _process_event(self, event: Event):
        """Process single event."""
        event.status = EventStatus.PROCESSING
        
        with self.lock:
            handlers = self.handlers.get(event.event_type, [])
            self.event_history.append(event)
        
        try:
            handled = False
            for handler in handlers:
                if handler.can_handle(event):
                    if handler.handle(event):
                        handled = True
                        break
            
            if handled:
                event.status = EventStatus.COMPLETED
                with self.lock:
                    self.processed_count += 1
            else:
                event.status = EventStatus.FAILED
                with self.lock:
                    self.failed_count += 1
        
        except Exception as e:
            event.status = EventStatus.FAILED
            event.metadata['error'] = str(e)
            with self.lock:
                self.failed_count += 1
    
    def get_statistics(self) -> Dict:
        """Get event bus statistics."""
        with self.lock:
            return {
                'total_events': self.event_count,
                'processed_events': self.processed_count,
                'failed_events': self.failed_count,
                'queue_size': self.event_queue.qsize(),
                'handlers_count': len(self.handlers),
                'is_running': self.is_running
            }
    
    def get_event_history(self, event_type: str = None, 
                         limit: int = 100) -> List[Event]:
        """Get event history."""
        with self.lock:
            if event_type:
                events = [e for e in self.event_history if e.event_type == event_type]
            else:
                events = self.event_history
            
            return events[-limit:]

class AsyncEventHandler(EventHandler):
    """Async event handler."""
    
    def __init__(self, event_types: Set[str], callback: Callable):
        self.event_types = event_types
        self.callback = callback
        self.handled_count = 0
    
    def can_handle(self, event: Event) -> bool:
        """Check if can handle."""
        return event.event_type in self.event_types
    
    def handle(self, event: Event) -> bool:
        """Handle event."""
        try:
            self.callback(event)
            self.handled_count += 1
            return True
        except Exception as e:
            print(f"Error handling event: {e}")
            return False

class EventAggregator:
    """Aggregate multiple events."""
    
    def __init__(self, event_bus: EventBus, window_seconds: int = 5):
        self.event_bus = event_bus
        self.window_seconds = window_seconds
        self.aggregated_events: Dict[str, List[Event]] = {}
        self.lock = threading.RLock()
        self.cleanup_thread = None
    
    def start(self):
        """Start aggregator."""
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True
        )
        self.cleanup_thread.start()
    
    def add_event(self, event: Event):
        """Add event to aggregation."""
        with self.lock:
            if event.event_type not in self.aggregated_events:
                self.aggregated_events[event.event_type] = []
            
            self.aggregated_events[event.event_type].append(event)
    
    def _cleanup_loop(self):
        """Cleanup old events."""
        while True:
            time.sleep(self.window_seconds)
            
            with self.lock:
                current_time = time.time()
                for event_type in self.aggregated_events:
                    self.aggregated_events[event_type] = [
                        e for e in self.aggregated_events[event_type]
                        if current_time - e.timestamp < self.window_seconds
                    ]
    
    def get_aggregated(self, event_type: str) -> List[Event]:
        """Get aggregated events."""
        with self.lock:
            return self.aggregated_events.get(event_type, []).copy()

class EventFilter:
    """Filter events based on criteria."""
    
    def __init__(self):
        self.filters: List[Callable] = []
    
    def add_filter(self, predicate: Callable) -> 'EventFilter':
        """Add filter predicate."""
        self.filters.append(predicate)
        return self
    
    def matches(self, event: Event) -> bool:
        """Check if event matches all filters."""
        return all(f(event) for f in self.filters)
    
    def filter_events(self, events: List[Event]) -> List[Event]:
        """Filter events."""
        return [e for e in events if self.matches(e)]

# Example usage
if __name__ == "__main__":
    # Create event bus
    bus = EventBus(max_workers=2)
    
    # Define handler
    class FaceDetectionHandler(AsyncEventHandler):
        def __init__(self):
            super().__init__({'face.detected'}, self.process)
        
        def process(self, event: Event):
            print(f"Processing face detection: {event.data}")
    
    # Subscribe
    handler = FaceDetectionHandler()
    bus.subscribe('face.detected', handler)
    
    # Start bus
    bus.start()
    
    # Publish events
    for i in range(5):
        event = Event(
            event_type='face.detected',
            source='camera_1',
            priority=EventPriority.HIGH,
            data={'face_id': f'face_{i}', 'confidence': 0.95}
        )
        bus.publish(event)
    
    time.sleep(2)
    
    # Statistics
    stats = bus.get_statistics()
    print(f"\nEvent Bus Statistics:")
    print(f"  Total Events: {stats['total_events']}")
    print(f"  Processed: {stats['processed_events']}")
    print(f"  Failed: {stats['failed_events']}")
    
    bus.stop()
