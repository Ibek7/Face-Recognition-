# Time-Series Database

import threading
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import math

class AggregationMethod(Enum):
    """Time-series aggregation methods."""
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    STDDEV = "stddev"

@dataclass
class DataPoint:
    """Time-series data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'tags': self.tags
        }

@dataclass
class TimeSeries:
    """Time-series data."""
    metric_name: str
    points: List[DataPoint] = field(default_factory=list)
    retention_days: int = 30
    created_at: float = field(default_factory=time.time)
    
    def add_point(self, point: DataPoint) -> None:
        """Add data point."""
        self.points.append(point)
        self.points.sort(key=lambda p: p.timestamp)

class TimeSeriesDB:
    """Time-series database."""
    
    def __init__(self):
        self.metrics: Dict[str, TimeSeries] = {}
        self.lock = threading.RLock()
        self.retention_policy: Dict[str, int] = {}
    
    def write(self, metric_name: str, value: float, timestamp: float = None,
             tags: Dict = None) -> None:
        """Write data point."""
        timestamp = timestamp or time.time()
        tags = tags or {}
        
        with self.lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = TimeSeries(metric_name)
            
            point = DataPoint(timestamp, value, tags)
            self.metrics[metric_name].add_point(point)
            
            # Enforce retention policy
            self._apply_retention(metric_name)
    
    def read(self, metric_name: str, start_time: float, end_time: float,
            tags_filter: Dict = None) -> List[DataPoint]:
        """Read data points."""
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            ts = self.metrics[metric_name]
            points = [p for p in ts.points
                     if start_time <= p.timestamp <= end_time]
            
            # Filter by tags
            if tags_filter:
                points = [p for p in points
                         if all(p.tags.get(k) == v for k, v in tags_filter.items())]
            
            return points
    
    def aggregate(self, metric_name: str, start_time: float, end_time: float,
                 method: AggregationMethod, window_sec: int = 60) -> List[Dict]:
        """Aggregate data points."""
        with self.lock:
            points = self.read(metric_name, start_time, end_time)
            
            if not points:
                return []
            
            # Group by time window
            windows: Dict[int, List[float]] = {}
            for point in points:
                window = int(point.timestamp / window_sec)
                if window not in windows:
                    windows[window] = []
                windows[window].append(point.value)
            
            # Aggregate each window
            results = []
            for window in sorted(windows.keys()):
                values = windows[window]
                window_timestamp = window * window_sec
                
                if method == AggregationMethod.MEAN:
                    agg_value = sum(values) / len(values)
                elif method == AggregationMethod.SUM:
                    agg_value = sum(values)
                elif method == AggregationMethod.MIN:
                    agg_value = min(values)
                elif method == AggregationMethod.MAX:
                    agg_value = max(values)
                elif method == AggregationMethod.COUNT:
                    agg_value = len(values)
                elif method == AggregationMethod.STDDEV:
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    agg_value = math.sqrt(variance)
                else:
                    agg_value = 0
                
                results.append({
                    'timestamp': window_timestamp,
                    'value': agg_value,
                    'count': len(values)
                })
            
            return results
    
    def downsample(self, metric_name: str, retention_days: int = 30,
                  target_interval: int = 3600) -> None:
        """Downsample old data."""
        with self.lock:
            if metric_name not in self.metrics:
                return
            
            ts = self.metrics[metric_name]
            cutoff_time = time.time() - (retention_days * 86400)
            
            # Keep recent points as-is, downsample older points
            old_points = [p for p in ts.points if p.timestamp < cutoff_time]
            recent_points = [p for p in ts.points if p.timestamp >= cutoff_time]
            
            # Aggregate old points
            windows: Dict[int, List[float]] = {}
            for point in old_points:
                window = int(point.timestamp / target_interval)
                if window not in windows:
                    windows[window] = []
                windows[window].append(point.value)
            
            # Create downsampled points
            downsampled = []
            for window in sorted(windows.keys()):
                values = windows[window]
                avg_value = sum(values) / len(values)
                downsampled.append(DataPoint(
                    timestamp=window * target_interval,
                    value=avg_value
                ))
            
            # Replace points
            ts.points = downsampled + recent_points
            ts.points.sort(key=lambda p: p.timestamp)
    
    def _apply_retention(self, metric_name: str) -> None:
        """Apply retention policy."""
        ts = self.metrics[metric_name]
        retention_seconds = ts.retention_days * 86400
        cutoff_time = time.time() - retention_seconds
        
        ts.points = [p for p in ts.points if p.timestamp >= cutoff_time]
    
    def delete_metric(self, metric_name: str) -> None:
        """Delete metric."""
        with self.lock:
            if metric_name in self.metrics:
                del self.metrics[metric_name]
    
    def get_metric_stats(self, metric_name: str) -> Dict:
        """Get metric statistics."""
        with self.lock:
            if metric_name not in self.metrics:
                return {}
            
            ts = self.metrics[metric_name]
            if not ts.points:
                return {'metric': metric_name, 'points': 0}
            
            values = [p.value for p in ts.points]
            return {
                'metric': metric_name,
                'points': len(ts.points),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'first_timestamp': ts.points[0].timestamp,
                'last_timestamp': ts.points[-1].timestamp
            }

class TimeSeriesIndex:
    """Index for efficient time-series queries."""
    
    def __init__(self):
        self.tag_index: Dict[str, Dict[str, List[str]]] = {}
        self.lock = threading.RLock()
    
    def index_metric(self, metric_name: str, tags: Dict) -> None:
        """Index metric by tags."""
        with self.lock:
            for key, value in tags.items():
                if key not in self.tag_index:
                    self.tag_index[key] = {}
                if value not in self.tag_index[key]:
                    self.tag_index[key][value] = []
                if metric_name not in self.tag_index[key][value]:
                    self.tag_index[key][value].append(metric_name)
    
    def find_metrics(self, tag_filters: Dict) -> List[str]:
        """Find metrics by tags."""
        with self.lock:
            results = None
            
            for key, value in tag_filters.items():
                if key in self.tag_index and value in self.tag_index[key]:
                    metrics = set(self.tag_index[key][value])
                    
                    if results is None:
                        results = metrics
                    else:
                        results = results.intersection(metrics)
            
            return list(results) if results else []

class RetentionPolicy:
    """Retention policy for time-series data."""
    
    def __init__(self, policy_id: str):
        self.policy_id = policy_id
        self.rules: List[Dict] = []
    
    def add_rule(self, duration_days: int, aggregation: str) -> None:
        """Add retention rule."""
        self.rules.append({
            'duration_days': duration_days,
            'aggregation': aggregation,
            'created_at': time.time()
        })
    
    def apply_rule(self, data: List[DataPoint], rule: Dict) -> List[DataPoint]:
        """Apply retention rule."""
        cutoff_time = time.time() - (rule['duration_days'] * 86400)
        return [p for p in data if p.timestamp >= cutoff_time]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'policy_id': self.policy_id,
            'rules': self.rules
        }

# Example usage
if __name__ == "__main__":
    db = TimeSeriesDB()
    
    # Write data points
    now = time.time()
    for i in range(100):
        db.write("cpu.usage", 50 + i * 0.5, now - (100 - i) * 60)
        db.write("memory.usage", 70 + i * 0.2, now - (100 - i) * 60)
    
    # Read recent data
    recent_points = db.read("cpu.usage", now - 3600, now)
    print(f"Recent CPU points: {len(recent_points)}")
    
    # Aggregate data
    aggregated = db.aggregate(
        "cpu.usage", 
        now - 3600, 
        now,
        AggregationMethod.MEAN,
        window_sec=300
    )
    print(f"\nAggregated CPU data (5-min windows):")
    for agg in aggregated[:3]:
        print(f"  {agg['timestamp']}: {agg['value']:.2f}")
    
    # Get statistics
    stats = db.get_metric_stats("cpu.usage")
    print(f"\nCPU Stats: {json.dumps(stats, indent=2)}")
    
    # Downsample
    db.downsample("cpu.usage", retention_days=30, target_interval=3600)
    print("\nDownsample completed")
