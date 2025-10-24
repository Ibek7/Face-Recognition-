# Distributed Tracing & Observability System

import threading
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

class SpanKind(Enum):
    """Type of span."""
    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"

class SpanStatus(Enum):
    """Span execution status."""
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"

@dataclass
class TraceContext:
    """Trace context for propagation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    flags: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'flags': self.flags
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'TraceContext':
        """Create from dictionary."""
        return TraceContext(
            trace_id=data.get('trace_id', str(uuid.uuid4())),
            span_id=data.get('span_id', str(uuid.uuid4())),
            parent_span_id=data.get('parent_span_id'),
            flags=data.get('flags', 0)
        )

@dataclass
class Event:
    """Span event."""
    name: str
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'timestamp': self.timestamp,
            'attributes': self.attributes
        }

@dataclass
class Span:
    """Distributed trace span."""
    trace_id: str
    span_id: str
    name: str
    kind: SpanKind
    start_time: float
    end_time: Optional[float] = None
    parent_span_id: Optional[str] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000
    
    def add_event(self, name: str, attributes: Dict[str, Any] = None) -> None:
        """Add event to span."""
        event = Event(name, time.time(), attributes or {})
        self.events.append(event)
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value
    
    def set_status(self, status: SpanStatus, description: str = "") -> None:
        """Set span status."""
        self.status = status
        if description:
            self.attributes['status.description'] = description
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'name': self.name,
            'kind': self.kind.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms(),
            'parent_span_id': self.parent_span_id,
            'status': self.status.value,
            'attributes': self.attributes,
            'events': [e.to_dict() for e in self.events],
            'links': self.links
        }

class Tracer:
    """Main tracer for creating spans."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.spans: Dict[str, Span] = {}
        self.current_context = None
        self.lock = threading.RLock()
    
    def start_span(self, name: str, kind: SpanKind = SpanKind.INTERNAL,
                   trace_id: Optional[str] = None,
                   parent_span_id: Optional[str] = None) -> Span:
        """Start a new span."""
        with self.lock:
            trace_id = trace_id or str(uuid.uuid4())
            span_id = str(uuid.uuid4())
            
            span = Span(
                trace_id=trace_id,
                span_id=span_id,
                name=name,
                kind=kind,
                start_time=time.time(),
                parent_span_id=parent_span_id,
                attributes={'service': self.service_name}
            )
            
            self.spans[span_id] = span
            self.current_context = TraceContext(trace_id, span_id, parent_span_id)
            
            return span
    
    def end_span(self, span_id: str, status: SpanStatus = SpanStatus.OK) -> None:
        """End a span."""
        with self.lock:
            if span_id in self.spans:
                span = self.spans[span_id]
                span.end_time = time.time()
                span.status = status
    
    def get_current_context(self) -> Optional[TraceContext]:
        """Get current trace context."""
        with self.lock:
            return self.current_context
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """Get span by ID."""
        with self.lock:
            return self.spans.get(span_id)
    
    def get_all_spans(self) -> List[Span]:
        """Get all spans."""
        with self.lock:
            return list(self.spans.values())

class TraceExporter:
    """Export traces to external systems."""
    
    def __init__(self):
        self.export_handlers: Dict[str, callable] = {}
        self.lock = threading.RLock()
    
    def register_handler(self, name: str, handler: callable) -> None:
        """Register export handler."""
        with self.lock:
            self.export_handlers[name] = handler
    
    def export_span(self, span: Span) -> None:
        """Export span."""
        with self.lock:
            for handler in self.export_handlers.values():
                try:
                    handler(span)
                except Exception as e:
                    print(f"Error exporting span: {e}")
    
    def export_spans(self, spans: List[Span]) -> None:
        """Export multiple spans."""
        for span in spans:
            self.export_span(span)

class TracingContext:
    """Thread-local tracing context."""
    
    _thread_local = threading.local()
    
    @classmethod
    def set_context(cls, context: TraceContext) -> None:
        """Set current context."""
        cls._thread_local.context = context
    
    @classmethod
    def get_context(cls) -> Optional[TraceContext]:
        """Get current context."""
        return getattr(cls._thread_local, 'context', None)
    
    @classmethod
    def clear_context(cls) -> None:
        """Clear current context."""
        cls._thread_local.context = None

class SpanDecorator:
    """Decorator for automatic span creation."""
    
    def __init__(self, tracer: Tracer, span_name: str = None, 
                 kind: SpanKind = SpanKind.INTERNAL):
        self.tracer = tracer
        self.span_name = span_name
        self.kind = kind
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            span_name = self.span_name or func.__name__
            
            # Get parent context
            parent_context = TracingContext.get_context()
            parent_span_id = parent_context.span_id if parent_context else None
            trace_id = parent_context.trace_id if parent_context else None
            
            # Create span
            span = self.tracer.start_span(
                span_name,
                kind=self.kind,
                trace_id=trace_id,
                parent_span_id=parent_span_id
            )
            
            # Set context
            context = TraceContext(span.trace_id, span.span_id, parent_span_id)
            TracingContext.set_context(context)
            
            try:
                result = func(*args, **kwargs)
                span.set_status(SpanStatus.OK)
                return result
            except Exception as e:
                span.set_status(SpanStatus.ERROR, str(e))
                raise
            finally:
                self.tracer.end_span(span.span_id, span.status)
                TracingContext.clear_context()
        
        return wrapper

class TraceAnalyzer:
    """Analyze traces for performance and errors."""
    
    @staticmethod
    def get_trace_tree(spans: List[Span]) -> Dict:
        """Build trace tree."""
        tree = {}
        span_map = {s.span_id: s for s in spans}
        
        # Find root spans
        roots = [s for s in spans if s.parent_span_id is None]
        
        for root in roots:
            tree[root.span_id] = TraceAnalyzer._build_tree_node(root, span_map)
        
        return tree
    
    @staticmethod
    def _build_tree_node(span: Span, span_map: Dict) -> Dict:
        """Build tree node."""
        children = [s for s in span_map.values() 
                   if s.parent_span_id == span.span_id]
        
        return {
            'span': span.to_dict(),
            'children': [TraceAnalyzer._build_tree_node(child, span_map)
                        for child in children]
        }
    
    @staticmethod
    def find_slow_spans(spans: List[Span], threshold_ms: float) -> List[Span]:
        """Find spans exceeding duration threshold."""
        return [s for s in spans 
               if s.duration_ms() and s.duration_ms() > threshold_ms]
    
    @staticmethod
    def find_errors(spans: List[Span]) -> List[Span]:
        """Find spans with errors."""
        return [s for s in spans if s.status == SpanStatus.ERROR]
    
    @staticmethod
    def get_statistics(spans: List[Span]) -> Dict:
        """Get trace statistics."""
        durations = [s.duration_ms() for s in spans if s.duration_ms()]
        
        return {
            'total_spans': len(spans),
            'error_count': len([s for s in spans if s.status == SpanStatus.ERROR]),
            'avg_duration_ms': sum(durations) / len(durations) if durations else 0,
            'max_duration_ms': max(durations) if durations else 0,
            'min_duration_ms': min(durations) if durations else 0
        }

# Example usage
if __name__ == "__main__":
    tracer = Tracer("face-recognition-service")
    exporter = TraceExporter()
    
    # Define export handler
    def console_exporter(span: Span):
        print(f"Span: {span.name} ({span.duration_ms():.2f}ms)")
    
    exporter.register_handler("console", console_exporter)
    
    # Create spans
    span1 = tracer.start_span("process_image", SpanKind.SERVER)
    span1.set_attribute("image_id", "img_123")
    span1.add_event("image_loaded")
    time.sleep(0.1)
    tracer.end_span(span1.span_id)
    exporter.export_span(span1)
    
    # Analyze
    all_spans = tracer.get_all_spans()
    stats = TraceAnalyzer.get_statistics(all_spans)
    print(f"\nTrace Statistics:")
    print(json.dumps(stats, indent=2))
