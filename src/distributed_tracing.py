# Distributed Tracing System

import uuid
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json

class SpanKind(Enum):
    """Span kind classification."""
    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"

class SpanStatus(Enum):
    """Span completion status."""
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"

@dataclass
class SpanContext:
    """Span context for correlation."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    is_remote: bool = False
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        return {
            'X-Trace-ID': self.trace_id,
            'X-Span-ID': self.span_id,
            'X-Parent-Span-ID': self.parent_span_id or ''
        }

@dataclass
class Event:
    """Span event."""
    name: str
    timestamp: float = field(default_factory=time.time)
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
    context: SpanContext
    name: str
    kind: SpanKind
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)
    links: List[Dict] = field(default_factory=list)
    resource: Dict[str, Any] = field(default_factory=dict)
    
    def add_attribute(self, key: str, value: Any):
        """Add span attribute."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """Add span event."""
        event = Event(name=name, attributes=attributes or {})
        self.events.append(event)
    
    def set_status(self, status: SpanStatus, description: str = None):
        """Set span status."""
        self.status = status
        if description:
            self.add_attribute('status.description', description)
    
    def end(self):
        """End span."""
        self.end_time = time.time()
    
    def get_duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'trace_id': self.context.trace_id,
            'span_id': self.context.span_id,
            'parent_span_id': self.context.parent_span_id,
            'name': self.name,
            'kind': self.kind.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.get_duration_ms(),
            'status': self.status.value,
            'attributes': self.attributes,
            'events': [event.to_dict() for event in self.events],
            'resource': self.resource
        }

class TracingContext:
    """Thread-local tracing context."""
    
    def __init__(self):
        self.local = threading.local()
    
    def set_span(self, span: Span):
        """Set current span."""
        self.local.span = span
    
    def get_span(self) -> Optional[Span]:
        """Get current span."""
        return getattr(self.local, 'span', None)
    
    def get_context(self) -> Optional[SpanContext]:
        """Get current span context."""
        span = self.get_span()
        return span.context if span else None

class Tracer:
    """Distributed tracer."""
    
    def __init__(self, service_name: str, service_version: str = "1.0.0"):
        self.service_name = service_name
        self.service_version = service_version
        self.spans: List[Span] = []
        self.context = TracingContext()
        self.lock = threading.RLock()
        self.exporters: List['SpanExporter'] = []
    
    def start_span(self, name: str, kind: SpanKind = SpanKind.INTERNAL,
                  attributes: Dict[str, Any] = None,
                  trace_id: str = None) -> Span:
        """Start new span."""
        
        parent_span = self.context.get_span()
        
        # Create or use existing trace
        if trace_id:
            trace_id = trace_id
        elif parent_span:
            trace_id = parent_span.context.trace_id
        else:
            trace_id = str(uuid.uuid4())
        
        # Create span context
        span_context = SpanContext(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span.context.span_id if parent_span else None
        )
        
        # Create span
        span = Span(
            context=span_context,
            name=name,
            kind=kind,
            attributes=attributes or {},
            resource={
                'service.name': self.service_name,
                'service.version': self.service_version
            }
        )
        
        # Store span
        with self.lock:
            self.spans.append(span)
        
        return span
    
    def activate_span(self, span: Span) -> 'SpanScope':
        """Activate span in context."""
        return SpanScope(self.context, span)
    
    def end_span(self, span: Span, status: SpanStatus = SpanStatus.OK):
        """End span."""
        span.set_status(status)
        span.end()
        
        # Export span
        with self.lock:
            for exporter in self.exporters:
                exporter.export_span(span)
    
    def add_exporter(self, exporter: 'SpanExporter'):
        """Add span exporter."""
        with self.lock:
            self.exporters.append(exporter)
    
    def get_spans(self, trace_id: str = None) -> List[Span]:
        """Get spans by trace ID."""
        with self.lock:
            if trace_id:
                return [s for s in self.spans if s.context.trace_id == trace_id]
            return self.spans.copy()
    
    def get_trace_summary(self, trace_id: str) -> Dict:
        """Get trace summary."""
        spans = self.get_spans(trace_id)
        
        if not spans:
            return None
        
        start_time = min(s.start_time for s in spans)
        end_time = max(s.end_time or time.time() for s in spans)
        
        return {
            'trace_id': trace_id,
            'span_count': len(spans),
            'start_time': start_time,
            'end_time': end_time,
            'duration_ms': (end_time - start_time) * 1000,
            'service': self.service_name,
            'status': self._get_overall_status(spans)
        }
    
    def _get_overall_status(self, spans: List[Span]) -> str:
        """Get overall trace status."""
        if any(s.status == SpanStatus.ERROR for s in spans):
            return SpanStatus.ERROR.value
        if all(s.status == SpanStatus.OK for s in spans):
            return SpanStatus.OK.value
        return SpanStatus.UNSET.value

class SpanScope:
    """Span activation scope."""
    
    def __init__(self, context: TracingContext, span: Span):
        self.context = context
        self.span = span
        self.previous_span = None
    
    def __enter__(self):
        """Enter scope."""
        self.previous_span = self.context.get_span()
        self.context.set_span(self.span)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit scope."""
        if exc_type:
            self.span.set_status(SpanStatus.ERROR, str(exc_val))
        
        self.span.end()
        self.context.set_span(self.previous_span)

class SpanExporter:
    """Base class for span exporters."""
    
    def export_span(self, span: Span):
        """Export span."""
        pass

class LogExporter(SpanExporter):
    """Log-based span exporter."""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.spans: List[Span] = []
        self.lock = threading.RLock()
    
    def export_span(self, span: Span):
        """Export span to log."""
        with self.lock:
            self.spans.append(span)
            
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(span.to_dict()) + '\n')

class MetricsCollector:
    """Collect metrics from spans."""
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = {}
        self.error_count: Dict[str, int] = {}
        self.lock = threading.RLock()
    
    def collect(self, span: Span):
        """Collect metrics from span."""
        with self.lock:
            operation = span.name
            
            # Track operation time
            if operation not in self.operation_times:
                self.operation_times[operation] = []
            self.operation_times[operation].append(span.get_duration_ms())
            
            # Track errors
            if span.status == SpanStatus.ERROR:
                self.error_count[operation] = self.error_count.get(operation, 0) + 1
    
    def get_stats(self, operation: str) -> Dict:
        """Get statistics for operation."""
        with self.lock:
            times = self.operation_times.get(operation, [])
            if not times:
                return None
            
            return {
                'operation': operation,
                'count': len(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'avg_ms': sum(times) / len(times),
                'errors': self.error_count.get(operation, 0)
            }

# Example usage
if __name__ == "__main__":
    # Create tracer
    tracer = Tracer("face-recognition-api", "1.0.0")
    
    # Add exporter
    log_exporter = LogExporter()
    tracer.add_exporter(log_exporter)
    
    # Create trace
    with tracer.activate_span(tracer.start_span("process_image", SpanKind.SERVER)) as scope:
        span = scope.span
        span.add_attribute("image.format", "jpg")
        
        # Simulate processing
        time.sleep(0.1)
        
        with tracer.activate_span(tracer.start_span("detect_faces", SpanKind.INTERNAL)) as nested_scope:
            nested_scope.span.add_event("detection_started")
            time.sleep(0.05)
            nested_scope.span.add_event("detection_completed", {"faces_found": 3})
            tracer.end_span(nested_scope.span, SpanStatus.OK)
        
        tracer.end_span(span, SpanStatus.OK)
    
    # Get trace summary
    trace_id = span.context.trace_id
    summary = tracer.get_trace_summary(trace_id)
    print(f"Trace Summary:")
    print(json.dumps(summary, indent=2))
