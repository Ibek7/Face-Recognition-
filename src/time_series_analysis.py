# Advanced Time-Series Analysis & Anomaly Detection

import threading
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics

class AnomalyType(Enum):
    """Types of anomalies."""
    POINT_ANOMALY = "point_anomaly"  # Single outlier point
    COLLECTIVE_ANOMALY = "collective_anomaly"  # Group of outliers
    CONTEXTUAL_ANOMALY = "contextual_anomaly"  # Unusual in context
    SEASONAL_ANOMALY = "seasonal_anomaly"  # Deviation from pattern

class DetectionMethod(Enum):
    """Anomaly detection methods."""
    ZSCORE = "zscore"
    IQR = "iqr"  # Interquartile range
    ISOLATION_FOREST = "isolation_forest"
    MOVING_AVERAGE = "moving_average"
    AUTOENCODER = "autoencoder"

@dataclass
class TimeSeriesPoint:
    """Single time-series data point."""
    timestamp: float
    value: float
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'label': self.label,
            'is_anomaly': self.is_anomaly,
            'anomaly_score': self.anomaly_score
        }

@dataclass
class AnomalyEvent:
    """Detected anomaly event."""
    event_id: str
    timestamp: float
    anomaly_type: AnomalyType
    severity: float  # 0.0 to 1.0
    description: str
    affected_points: List[float]
    method_used: DetectionMethod
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'type': self.anomaly_type.value,
            'severity': self.severity,
            'description': self.description,
            'affected_points': self.affected_points,
            'method': self.method_used.value
        }

class TimeSeriesBuffer:
    """Buffer for time-series data."""
    
    def __init__(self, max_size: int = 1000, retention_sec: int = 3600):
        self.max_size = max_size
        self.retention_sec = retention_sec
        self.data: deque = deque(maxlen=max_size)
        self.lock = threading.RLock()
    
    def add_point(self, point: TimeSeriesPoint) -> None:
        """Add data point."""
        with self.lock:
            self.data.append(point)
            self._cleanup_old_data()
    
    def _cleanup_old_data(self) -> None:
        """Remove data older than retention."""
        cutoff_time = time.time() - self.retention_sec
        
        while self.data and self.data[0].timestamp < cutoff_time:
            self.data.popleft()
    
    def get_points(self, start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> List[TimeSeriesPoint]:
        """Get points in time range."""
        with self.lock:
            points = list(self.data)
        
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]
        
        return points
    
    def get_latest(self, count: int = 100) -> List[TimeSeriesPoint]:
        """Get latest N points."""
        with self.lock:
            return list(deque(self.data, maxlen=count))

class StatisticalAnomalyDetector:
    """Detect anomalies using statistical methods."""
    
    def __init__(self, zscore_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
    
    def detect_zscore(self, points: List[TimeSeriesPoint]) -> List[Tuple[TimeSeriesPoint, float]]:
        """Detect anomalies using Z-score."""
        if len(points) < 2:
            return []
        
        values = [p.value for p in points]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        anomalies = []
        
        for point in points:
            if stdev > 0:
                zscore = abs((point.value - mean) / stdev)
                
                if zscore > self.zscore_threshold:
                    anomalies.append((point, zscore / self.zscore_threshold))
        
        return anomalies
    
    def detect_iqr(self, points: List[TimeSeriesPoint]) -> List[Tuple[TimeSeriesPoint, float]]:
        """Detect anomalies using IQR method."""
        if len(points) < 4:
            return []
        
        values = sorted([p.value for p in points])
        q1_idx = len(values) // 4
        q3_idx = 3 * len(values) // 4
        
        q1 = values[q1_idx]
        q3 = values[q3_idx]
        iqr = q3 - q1
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        anomalies = []
        
        for point in points:
            if point.value < lower_bound or point.value > upper_bound:
                distance = max(
                    lower_bound - point.value,
                    point.value - upper_bound
                )
                anomalies.append((point, distance / (iqr if iqr > 0 else 1)))
        
        return anomalies

class MovingAverageDetector:
    """Detect anomalies using moving averages."""
    
    def __init__(self, window_size: int = 5, deviation_threshold: float = 2.0):
        self.window_size = window_size
        self.deviation_threshold = deviation_threshold
    
    def detect(self, points: List[TimeSeriesPoint]) -> List[Tuple[TimeSeriesPoint, float]]:
        """Detect anomalies using moving average."""
        if len(points) < self.window_size:
            return []
        
        anomalies = []
        
        for i in range(self.window_size, len(points)):
            window = [p.value for p in points[i - self.window_size:i]]
            moving_avg = statistics.mean(window)
            moving_stdev = statistics.stdev(window) if len(window) > 1 else 0
            
            current_value = points[i].value
            deviation = abs(current_value - moving_avg)
            
            if moving_stdev > 0 and deviation > self.deviation_threshold * moving_stdev:
                anomalies.append((points[i], deviation / moving_stdev))
        
        return anomalies

class SeasonalityAnalyzer:
    """Analyze seasonality in time series."""
    
    def __init__(self, season_length: int = 24):
        self.season_length = season_length
        self.seasonal_pattern: List[float] = []
        self.lock = threading.RLock()
    
    def extract_seasonal_pattern(self, points: List[TimeSeriesPoint]) -> List[float]:
        """Extract seasonal pattern."""
        if len(points) < self.season_length * 2:
            return []
        
        # Calculate average for each seasonal position
        seasonal = [[] for _ in range(self.season_length)]
        
        for i, point in enumerate(points):
            seasonal[i % self.season_length].append(point.value)
        
        with self.lock:
            self.seasonal_pattern = [
                statistics.mean(values) if values else 0
                for values in seasonal
            ]
        
        return self.seasonal_pattern
    
    def detect_seasonal_anomalies(self, points: List[TimeSeriesPoint],
                                 threshold: float = 2.0) -> List[Tuple[TimeSeriesPoint, float]]:
        """Detect anomalies from seasonal pattern."""
        with self.lock:
            if not self.seasonal_pattern:
                return []
            
            anomalies = []
            
            for i, point in enumerate(points[-self.season_length:]):
                seasonal_value = self.seasonal_pattern[i % self.season_length]
                deviation = abs(point.value - seasonal_value)
                
                if deviation > threshold:
                    anomalies.append((point, deviation / threshold))
            
            return anomalies

class AnomalyDetectionEngine:
    """Comprehensive anomaly detection."""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer = TimeSeriesBuffer(max_size=buffer_size)
        self.statistical_detector = StatisticalAnomalyDetector()
        self.moving_avg_detector = MovingAverageDetector()
        self.seasonality_analyzer = SeasonalityAnalyzer()
        
        self.detected_anomalies: List[AnomalyEvent] = []
        self.lock = threading.RLock()
    
    def add_point(self, point: TimeSeriesPoint) -> None:
        """Add data point."""
        self.buffer.add_point(point)
    
    def detect_anomalies(self, method: DetectionMethod = DetectionMethod.ZSCORE) -> List[AnomalyEvent]:
        """Detect anomalies using specified method."""
        points = self.buffer.get_latest(100)
        
        if not points:
            return []
        
        anomaly_candidates = []
        
        if method == DetectionMethod.ZSCORE:
            anomaly_candidates = self.statistical_detector.detect_zscore(points)
        elif method == DetectionMethod.IQR:
            anomaly_candidates = self.statistical_detector.detect_iqr(points)
        elif method == DetectionMethod.MOVING_AVERAGE:
            anomaly_candidates = self.moving_avg_detector.detect(points)
        elif method == DetectionMethod.SEASONAL_ANOMALY:
            anomaly_candidates = self.seasonality_analyzer.detect_seasonal_anomalies(points)
        
        # Convert to events
        events = []
        
        for point, score in anomaly_candidates:
            event = AnomalyEvent(
                event_id=self._generate_id(),
                timestamp=point.timestamp,
                anomaly_type=AnomalyType.POINT_ANOMALY,
                severity=min(score / 5.0, 1.0),
                description=f"Anomaly detected with score {score:.2f}",
                affected_points=[point.value],
                method_used=method
            )
            
            events.append(event)
            point.is_anomaly = True
            point.anomaly_score = score
        
        with self.lock:
            self.detected_anomalies.extend(events)
        
        return events
    
    def get_anomalies(self, time_range: Tuple[float, float] = None) -> List[AnomalyEvent]:
        """Get detected anomalies."""
        with self.lock:
            anomalies = list(self.detected_anomalies)
        
        if time_range:
            start, end = time_range
            anomalies = [
                a for a in anomalies
                if start <= a.timestamp <= end
            ]
        
        return anomalies
    
    def get_statistics(self) -> Dict:
        """Get time-series statistics."""
        points = self.buffer.get_latest()
        
        if not points:
            return {}
        
        values = [p.value for p in points]
        anomaly_count = sum(1 for p in points if p.is_anomaly)
        
        return {
            'point_count': len(points),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_count / len(points) if points else 0
        }
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        import hashlib
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

class TrendAnalyzer:
    """Analyze trends in time series."""
    
    def __init__(self):
        self.trend_history: List[float] = []
        self.lock = threading.RLock()
    
    def calculate_trend(self, points: List[TimeSeriesPoint]) -> Tuple[float, str]:
        """Calculate trend and direction."""
        if len(points) < 2:
            return 0.0, "flat"
        
        values = [p.value for p in points]
        
        # Simple linear regression trend
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        
        numerator = sum(
            (i - x_mean) * (values[i] - y_mean)
            for i in range(n)
        )
        
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        trend = numerator / denominator if denominator > 0 else 0
        
        if trend > 0.01:
            direction = "uptrend"
        elif trend < -0.01:
            direction = "downtrend"
        else:
            direction = "flat"
        
        with self.lock:
            self.trend_history.append(trend)
        
        return trend, direction

# Example usage
if __name__ == "__main__":
    # Create engine
    engine = AnomalyDetectionEngine()
    
    # Add sample points
    current_time = time.time()
    for i in range(100):
        point = TimeSeriesPoint(
            timestamp=current_time + i,
            value=50 + (10 * (i % 10)) + (2 if i % 7 == 0 else 0)
        )
        engine.add_point(point)
    
    # Add anomaly
    anomaly_point = TimeSeriesPoint(
        timestamp=current_time + 100,
        value=150  # Obvious spike
    )
    engine.add_point(anomaly_point)
    
    # Detect anomalies
    anomalies = engine.detect_anomalies(DetectionMethod.ZSCORE)
    print(f"Anomalies detected: {len(anomalies)}")
    for anomaly in anomalies:
        print(json.dumps(anomaly.to_dict(), indent=2, default=str))
    
    # Get statistics
    stats = engine.get_statistics()
    print(f"\nTime-Series Statistics:")
    print(json.dumps(stats, indent=2, default=str))
