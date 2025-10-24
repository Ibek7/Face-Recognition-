# Real-time Pub/Sub Messaging System

import threading
import time
import json
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import queue

class MessageType(Enum):
    """Message types."""
    EVENT = "event"
    COMMAND = "command"
    QUERY = "query"
    RESPONSE = "response"

class SubscriptionMode(Enum):
    """Subscription modes."""
    SYNC = "sync"  # Immediate delivery
    ASYNC = "async"  # Queue-based delivery
    BATCH = "batch"  # Batched delivery

@dataclass
class Message:
    """Pub/Sub message."""
    message_id: str
    topic: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    sender_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'message_id': self.message_id,
            'topic': self.topic,
            'type': self.message_type.value,
            'timestamp': self.timestamp,
            'sender': self.sender_id
        }

@dataclass
class Subscription:
    """Message subscription."""
    subscription_id: str
    topic_pattern: str  # Can use wildcards
    handler: Callable
    mode: SubscriptionMode = SubscriptionMode.ASYNC
    filter_func: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    message_count: int = 0
    is_active: bool = True
    
    def matches_topic(self, topic: str) -> bool:
        """Check if subscription matches topic."""
        if self.topic_pattern == "*":
            return True
        if self.topic_pattern == topic:
            return True
        # Simple wildcard support
        if "*" in self.topic_pattern:
            import fnmatch
            return fnmatch.fnmatch(topic, self.topic_pattern)
        return False

class MessageBroker:
    """Central message broker for pub/sub."""
    
    def __init__(self, broker_id: str):
        self.broker_id = broker_id
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self.message_queue: queue.Queue = queue.Queue()
        self.message_history: Dict[str, List[Message]] = defaultdict(list)
        self.dead_letter_queue: List[Message] = []
        
        self.is_running = False
        self.lock = threading.RLock()
    
    def subscribe(self, topic: str, handler: Callable,
                 mode: SubscriptionMode = SubscriptionMode.ASYNC,
                 filter_func: Callable = None) -> str:
        """Subscribe to topic."""
        subscription_id = f"sub_{len(self.subscriptions)}"
        
        subscription = Subscription(
            subscription_id=subscription_id,
            topic_pattern=topic,
            handler=handler,
            mode=mode,
            filter_func=filter_func
        )
        
        with self.lock:
            self.subscriptions[topic].append(subscription)
        
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from topic."""
        with self.lock:
            for topic, subs in self.subscriptions.items():
                for sub in subs:
                    if sub.subscription_id == subscription_id:
                        sub.is_active = False
                        return True
        
        return False
    
    def publish(self, topic: str, payload: Dict[str, Any],
               message_type: MessageType = MessageType.EVENT,
               sender_id: str = None) -> str:
        """Publish message to topic."""
        message_id = f"msg_{int(time.time() * 1000)}"
        
        message = Message(
            message_id=message_id,
            topic=topic,
            message_type=message_type,
            payload=payload,
            sender_id=sender_id
        )
        
        # Store in history
        with self.lock:
            self.message_history[topic].append(message)
            # Keep only last 1000 messages
            if len(self.message_history[topic]) > 1000:
                self.message_history[topic].pop(0)
        
        # Queue for delivery
        self.message_queue.put(message)
        
        return message_id
    
    def start(self, num_workers: int = 2) -> None:
        """Start message broker."""
        self.is_running = True
        
        for _ in range(num_workers):
            thread = threading.Thread(target=self._delivery_worker)
            thread.daemon = True
            thread.start()
    
    def stop(self) -> None:
        """Stop message broker."""
        self.is_running = False
    
    def _delivery_worker(self) -> None:
        """Worker thread for message delivery."""
        while self.is_running:
            try:
                message = self.message_queue.get(timeout=1.0)
                self._deliver_message(message)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in delivery worker: {e}")
    
    def _deliver_message(self, message: Message) -> None:
        """Deliver message to subscribers."""
        with self.lock:
            # Find matching subscriptions
            matching_subs = []
            
            for topic, subs in self.subscriptions.items():
                for sub in subs:
                    if sub.is_active and sub.matches_topic(message.topic):
                        matching_subs.append(sub)
        
        # Deliver to each subscriber
        for sub in matching_subs:
            try:
                # Apply filter
                if sub.filter_func and not sub.filter_func(message):
                    continue
                
                # Deliver based on mode
                if sub.mode == SubscriptionMode.SYNC:
                    sub.handler(message)
                else:  # ASYNC or BATCH
                    sub.handler(message)
                
                sub.message_count += 1
            
            except Exception as e:
                print(f"Error delivering message to subscriber: {e}")
                self.dead_letter_queue.append(message)
    
    def get_status(self) -> Dict:
        """Get broker status."""
        with self.lock:
            total_subs = sum(len(subs) for subs in self.subscriptions.values())
            active_subs = sum(
                len([s for s in subs if s.is_active])
                for subs in self.subscriptions.values()
            )
            
            return {
                'broker_id': self.broker_id,
                'is_running': self.is_running,
                'total_subscriptions': total_subs,
                'active_subscriptions': active_subs,
                'topics': len(self.subscriptions),
                'queue_size': self.message_queue.qsize(),
                'dead_letters': len(self.dead_letter_queue)
            }

class TopicManager:
    """Manage topics and routing."""
    
    def __init__(self):
        self.topics: Dict[str, Dict] = {}
        self.topic_hierarchy: Dict[str, List[str]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def register_topic(self, topic: str, description: str = "",
                      schema: Dict = None) -> None:
        """Register topic."""
        with self.lock:
            self.topics[topic] = {
                'description': description,
                'schema': schema or {},
                'created_at': time.time(),
                'message_count': 0
            }
    
    def create_topic_hierarchy(self, parent: str, child: str) -> None:
        """Create topic hierarchy."""
        with self.lock:
            self.topic_hierarchy[parent].append(child)
    
    def get_subtopics(self, topic: str) -> List[str]:
        """Get subtopics."""
        with self.lock:
            return self.topic_hierarchy.get(topic, [])

class RequestReplyPattern:
    """Implement request-reply messaging pattern."""
    
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.pending_replies: Dict[str, Any] = {}
        self.reply_timeout = 5.0
        self.lock = threading.RLock()
    
    def request(self, topic: str, payload: Dict, timeout: float = 5.0) -> Optional[Any]:
        """Send request and wait for reply."""
        request_id = f"req_{int(time.time() * 1000)}"
        
        # Subscribe to reply topic
        reply_topic = f"{topic}_reply_{request_id}"
        
        reply_event = threading.Event()
        reply_data = {}
        
        def reply_handler(message: Message):
            reply_data['response'] = message.payload
            reply_event.set()
        
        sub_id = self.broker.subscribe(reply_topic, reply_handler)
        
        # Send request
        self.broker.publish(
            topic,
            payload,
            message_type=MessageType.QUERY,
            sender_id=request_id
        )
        
        # Wait for reply
        if reply_event.wait(timeout=timeout):
            self.broker.unsubscribe(sub_id)
            return reply_data.get('response')
        
        # Timeout
        self.broker.unsubscribe(sub_id)
        return None

class EventBus:
    """Higher-level event bus using pub/sub."""
    
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def on(self, event_type: str, handler: Callable) -> None:
        """Register event handler."""
        with self.lock:
            self.event_handlers[event_type].append(handler)
            
            # Subscribe to broker
            self.broker.subscribe(
                event_type,
                lambda msg: handler(msg.payload)
            )
    
    def emit(self, event_type: str, data: Dict) -> None:
        """Emit event."""
        self.broker.publish(event_type, data, MessageType.EVENT)

# Example usage
if __name__ == "__main__":
    # Create broker
    broker = MessageBroker("broker1")
    broker.start(num_workers=2)
    
    # Subscribe to topics
    def face_detection_handler(message: Message):
        print(f"Face detection event: {message.payload}")
    
    def process_complete_handler(message: Message):
        print(f"Process complete: {message.payload}")
    
    broker.subscribe("face.detect", face_detection_handler)
    broker.subscribe("process.*", process_complete_handler)
    
    # Publish messages
    time.sleep(0.1)
    broker.publish("face.detect", {"image": "test.jpg", "confidence": 0.95})
    broker.publish("process.complete", {"status": "success"})
    
    # Wait for delivery
    time.sleep(1)
    
    # Get status
    status = broker.get_status()
    print("\nBroker Status:")
    print(json.dumps(status, indent=2))
    
    broker.stop()
