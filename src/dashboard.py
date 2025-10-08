# System Dashboard for Face Recognition System

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import psutil
import logging
from collections import deque, defaultdict
import numpy as np

@dataclass
class SystemMetrics:
    """System metrics data structure."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    requests_per_minute: int
    error_rate: float
    response_time_avg: float
    queue_length: int
    cache_hit_rate: float

@dataclass
class FaceRecognitionMetrics:
    """Face recognition specific metrics."""
    total_recognitions: int
    successful_recognitions: int
    failed_recognitions: int
    average_confidence: float
    processing_time_avg: float
    faces_detected_total: int
    unique_persons_recognized: int
    recognition_accuracy: float

class SystemDashboard:
    """Real-time system dashboard for monitoring face recognition system."""
    
    def __init__(self):
        self.app = FastAPI(title="Face Recognition System Dashboard")
        self.connected_clients: List[WebSocket] = []
        
        # Metrics storage
        self.system_metrics_history = deque(maxlen=1000)
        self.face_metrics_history = deque(maxlen=1000)
        self.alerts_history = deque(maxlen=100)
        
        # Performance tracking
        self.request_times = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0
        
        # Recognition tracking
        self.recognition_stats = defaultdict(int)
        self.confidence_scores = deque(maxlen=1000)
        self.processing_times = deque(maxlen=1000)
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_usage": 90.0,
            "memory_usage": 90.0,
            "error_rate": 0.05,
            "response_time": 5.0,
            "queue_length": 100
        }
        
        # Monitoring state
        self.monitoring_active = True
        
        self._setup_routes()
        self._start_monitoring()
    
    def _setup_routes(self):
        """Setup FastAPI routes for dashboard."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            return self._get_dashboard_html()
        
        @self.app.get("/api/metrics/system")
        async def get_system_metrics():
            return self._get_current_system_metrics()
        
        @self.app.get("/api/metrics/face-recognition")
        async def get_face_metrics():
            return self._get_current_face_metrics()
        
        @self.app.get("/api/metrics/history")
        async def get_metrics_history(hours: int = 1):
            return self._get_metrics_history(hours)
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            return list(self.alerts_history)
        
        @self.app.get("/api/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.websocket("/ws/metrics")
        async def websocket_metrics(websocket: WebSocket):
            await self._handle_websocket(websocket)
        
        @self.app.post("/api/alerts/configure")
        async def configure_alerts(thresholds: Dict[str, float]):
            self.alert_thresholds.update(thresholds)
            return {"status": "updated", "thresholds": self.alert_thresholds}
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connections for real-time updates."""
        await websocket.accept()
        self.connected_clients.append(websocket)
        
        try:
            while True:
                # Send real-time metrics every second
                metrics_data = {
                    "system": self._get_current_system_metrics(),
                    "face_recognition": self._get_current_face_metrics(),
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send_json(metrics_data)
                await asyncio.sleep(1)
                
        except WebSocketDisconnect:
            self.connected_clients.remove(websocket)
        except Exception as e:
            logging.error(f"WebSocket error: {e}")
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)
    
    def _start_monitoring(self):
        """Start background monitoring tasks."""
        asyncio.create_task(self._monitor_system_metrics())
        asyncio.create_task(self._check_alerts())
    
    async def _monitor_system_metrics(self):
        """Continuously monitor system metrics."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                # Calculate derived metrics
                error_rate = self.error_count / max(self.total_requests, 1)
                avg_response_time = np.mean(list(self.request_times)) if self.request_times else 0
                
                metrics = SystemMetrics(
                    timestamp=datetime.now(),
                    cpu_usage=cpu_percent,
                    memory_usage=memory.percent,
                    disk_usage=(disk.used / disk.total) * 100,
                    network_io={
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv
                    },
                    active_connections=len(self.connected_clients),
                    requests_per_minute=len(self.request_times),
                    error_rate=error_rate,
                    response_time_avg=avg_response_time,
                    queue_length=0,  # Would be updated by actual queue monitoring
                    cache_hit_rate=0  # Would be updated by cache monitoring
                )
                
                self.system_metrics_history.append(metrics)
                
                # Broadcast to connected clients
                await self._broadcast_metrics_update(metrics)
                
            except Exception as e:
                logging.error(f"System monitoring error: {e}")
            
            await asyncio.sleep(5)  # Monitor every 5 seconds
    
    async def _check_alerts(self):
        """Check for alert conditions."""
        while self.monitoring_active:
            try:
                if self.system_metrics_history:
                    latest_metrics = self.system_metrics_history[-1]
                    
                    # Check each threshold
                    if latest_metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
                        await self._create_alert("high_cpu", f"CPU usage: {latest_metrics.cpu_usage:.1f}%")
                    
                    if latest_metrics.memory_usage > self.alert_thresholds["memory_usage"]:
                        await self._create_alert("high_memory", f"Memory usage: {latest_metrics.memory_usage:.1f}%")
                    
                    if latest_metrics.error_rate > self.alert_thresholds["error_rate"]:
                        await self._create_alert("high_error_rate", f"Error rate: {latest_metrics.error_rate:.2%}")
                    
                    if latest_metrics.response_time_avg > self.alert_thresholds["response_time"]:
                        await self._create_alert("slow_response", f"Response time: {latest_metrics.response_time_avg:.2f}s")
                
            except Exception as e:
                logging.error(f"Alert checking error: {e}")
            
            await asyncio.sleep(30)  # Check alerts every 30 seconds
    
    async def _create_alert(self, alert_type: str, message: str):
        """Create and broadcast an alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "severity": self._get_alert_severity(alert_type)
        }
        
        self.alerts_history.append(alert)
        
        # Broadcast alert to connected clients
        alert_data = {"type": "alert", "data": alert}
        await self._broadcast_to_clients(alert_data)
        
        logging.warning(f"Alert: {alert_type} - {message}")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Determine alert severity."""
        critical_alerts = ["high_memory", "high_error_rate"]
        warning_alerts = ["high_cpu", "slow_response"]
        
        if alert_type in critical_alerts:
            return "critical"
        elif alert_type in warning_alerts:
            return "warning"
        else:
            return "info"
    
    async def _broadcast_metrics_update(self, metrics: SystemMetrics):
        """Broadcast metrics update to all connected clients."""
        update_data = {
            "type": "metrics_update",
            "data": asdict(metrics)
        }
        await self._broadcast_to_clients(update_data)
    
    async def _broadcast_to_clients(self, data: Dict):
        """Broadcast data to all connected WebSocket clients."""
        if not self.connected_clients:
            return
        
        disconnected_clients = []
        
        for client in self.connected_clients:
            try:
                await client.send_json(data)
            except Exception:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.remove(client)
    
    def _get_current_system_metrics(self) -> Dict:
        """Get current system metrics."""
        if not self.system_metrics_history:
            return {}
        
        latest = self.system_metrics_history[-1]
        return asdict(latest)
    
    def _get_current_face_metrics(self) -> Dict:
        """Get current face recognition metrics."""
        total_recognitions = self.recognition_stats["total"]
        successful = self.recognition_stats["successful"]
        failed = self.recognition_stats["failed"]
        
        metrics = FaceRecognitionMetrics(
            total_recognitions=total_recognitions,
            successful_recognitions=successful,
            failed_recognitions=failed,
            average_confidence=np.mean(list(self.confidence_scores)) if self.confidence_scores else 0,
            processing_time_avg=np.mean(list(self.processing_times)) if self.processing_times else 0,
            faces_detected_total=self.recognition_stats["faces_detected"],
            unique_persons_recognized=self.recognition_stats["unique_persons"],
            recognition_accuracy=successful / max(total_recognitions, 1)
        )
        
        return asdict(metrics)
    
    def _get_metrics_history(self, hours: int) -> Dict:
        """Get historical metrics for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics within time range
        recent_system_metrics = [
            asdict(m) for m in self.system_metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        return {
            "system_metrics": recent_system_metrics,
            "time_range_hours": hours,
            "total_data_points": len(recent_system_metrics)
        }
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Face Recognition System Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5; 
                }
                .header { 
                    background: #2c3e50; 
                    color: white; 
                    padding: 20px; 
                    margin: -20px -20px 20px -20px; 
                    text-align: center; 
                }
                .metrics-grid { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; 
                    margin-bottom: 20px; 
                }
                .metric-card { 
                    background: white; 
                    padding: 20px; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                }
                .metric-value { 
                    font-size: 2em; 
                    font-weight: bold; 
                    color: #3498db; 
                }
                .metric-label { 
                    color: #7f8c8d; 
                    margin-top: 5px; 
                }
                .alert { 
                    padding: 10px; 
                    margin: 10px 0; 
                    border-radius: 4px; 
                    border-left: 4px solid; 
                }
                .alert-critical { 
                    background: #ffebee; 
                    border-color: #f44336; 
                    color: #c62828; 
                }
                .alert-warning { 
                    background: #fff3e0; 
                    border-color: #ff9800; 
                    color: #ef6c00; 
                }
                .chart-container { 
                    background: white; 
                    padding: 20px; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                    margin-bottom: 20px; 
                }
                .status-indicator { 
                    width: 12px; 
                    height: 12px; 
                    border-radius: 50%; 
                    display: inline-block; 
                    margin-right: 8px; 
                }
                .status-healthy { background-color: #4CAF50; }
                .status-warning { background-color: #FF9800; }
                .status-critical { background-color: #F44336; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Face Recognition System Dashboard</h1>
                <div id="system-status">
                    <span class="status-indicator status-healthy"></span>
                    <span>System Status: </span>
                    <span id="status-text">Healthy</span>
                </div>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="cpu-usage">0%</div>
                    <div class="metric-label">CPU Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="memory-usage">0%</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="response-time">0ms</div>
                    <div class="metric-label">Avg Response Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="requests-per-minute">0</div>
                    <div class="metric-label">Requests/Min</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="recognition-accuracy">0%</div>
                    <div class="metric-label">Recognition Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="total-recognitions">0</div>
                    <div class="metric-label">Total Recognitions</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>System Performance</h3>
                <div id="performance-chart" style="height: 400px;"></div>
            </div>
            
            <div class="chart-container">
                <h3>Face Recognition Metrics</h3>
                <div id="recognition-chart" style="height: 400px;"></div>
            </div>
            
            <div class="chart-container">
                <h3>Recent Alerts</h3>
                <div id="alerts-container"></div>
            </div>
            
            <script>
                const ws = new WebSocket(`ws://${window.location.host}/ws/metrics`);
                
                // Initialize charts
                const performanceChart = document.getElementById('performance-chart');
                const recognitionChart = document.getElementById('recognition-chart');
                
                let performanceData = {
                    cpu: {x: [], y: [], name: 'CPU Usage', type: 'scatter'},
                    memory: {x: [], y: [], name: 'Memory Usage', type: 'scatter'},
                    response_time: {x: [], y: [], name: 'Response Time', type: 'scatter', yaxis: 'y2'}
                };
                
                let recognitionData = {
                    accuracy: {x: [], y: [], name: 'Accuracy', type: 'scatter'},
                    processing_time: {x: [], y: [], name: 'Processing Time', type: 'scatter', yaxis: 'y2'}
                };
                
                // Chart layouts
                const performanceLayout = {
                    title: 'System Performance Over Time',
                    xaxis: {title: 'Time'},
                    yaxis: {title: 'Percentage (%)', range: [0, 100]},
                    yaxis2: {title: 'Response Time (ms)', overlaying: 'y', side: 'right'}
                };
                
                const recognitionLayout = {
                    title: 'Face Recognition Performance',
                    xaxis: {title: 'Time'},
                    yaxis: {title: 'Accuracy (%)'},
                    yaxis2: {title: 'Processing Time (ms)', overlaying: 'y', side: 'right'}
                };
                
                Plotly.newPlot(performanceChart, Object.values(performanceData), performanceLayout);
                Plotly.newPlot(recognitionChart, Object.values(recognitionData), recognitionLayout);
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'metrics_update') {
                        updateMetrics(data.data);
                    } else if (data.type === 'alert') {
                        addAlert(data.data);
                    } else if (data.system && data.face_recognition) {
                        updateDashboard(data);
                    }
                };
                
                function updateDashboard(data) {
                    const system = data.system;
                    const recognition = data.face_recognition;
                    
                    // Update metric cards
                    document.getElementById('cpu-usage').textContent = system.cpu_usage.toFixed(1) + '%';
                    document.getElementById('memory-usage').textContent = system.memory_usage.toFixed(1) + '%';
                    document.getElementById('response-time').textContent = (system.response_time_avg * 1000).toFixed(0) + 'ms';
                    document.getElementById('requests-per-minute').textContent = system.requests_per_minute;
                    document.getElementById('recognition-accuracy').textContent = (recognition.recognition_accuracy * 100).toFixed(1) + '%';
                    document.getElementById('total-recognitions').textContent = recognition.total_recognitions;
                    
                    // Update system status
                    updateSystemStatus(system);
                    
                    // Update charts
                    updateCharts(data.timestamp, system, recognition);
                }
                
                function updateSystemStatus(system) {
                    const statusElement = document.getElementById('status-text');
                    const indicator = document.querySelector('.status-indicator');
                    
                    let status = 'Healthy';
                    let className = 'status-healthy';
                    
                    if (system.cpu_usage > 90 || system.memory_usage > 90) {
                        status = 'Critical';
                        className = 'status-critical';
                    } else if (system.cpu_usage > 80 || system.memory_usage > 80) {
                        status = 'Warning';
                        className = 'status-warning';
                    }
                    
                    statusElement.textContent = status;
                    indicator.className = 'status-indicator ' + className;
                }
                
                function updateCharts(timestamp, system, recognition) {
                    const time = new Date(timestamp);
                    
                    // Update performance chart
                    performanceData.cpu.x.push(time);
                    performanceData.cpu.y.push(system.cpu_usage);
                    
                    performanceData.memory.x.push(time);
                    performanceData.memory.y.push(system.memory_usage);
                    
                    performanceData.response_time.x.push(time);
                    performanceData.response_time.y.push(system.response_time_avg * 1000);
                    
                    // Keep only last 50 points
                    Object.values(performanceData).forEach(trace => {
                        if (trace.x.length > 50) {
                            trace.x.shift();
                            trace.y.shift();
                        }
                    });
                    
                    // Update recognition chart
                    recognitionData.accuracy.x.push(time);
                    recognitionData.accuracy.y.push(recognition.recognition_accuracy * 100);
                    
                    recognitionData.processing_time.x.push(time);
                    recognitionData.processing_time.y.push(recognition.processing_time_avg * 1000);
                    
                    // Keep only last 50 points
                    Object.values(recognitionData).forEach(trace => {
                        if (trace.x.length > 50) {
                            trace.x.shift();
                            trace.y.shift();
                        }
                    });
                    
                    // Redraw charts
                    Plotly.redraw(performanceChart);
                    Plotly.redraw(recognitionChart);
                }
                
                function addAlert(alert) {
                    const container = document.getElementById('alerts-container');
                    const alertDiv = document.createElement('div');
                    alertDiv.className = `alert alert-${alert.severity}`;
                    alertDiv.innerHTML = `
                        <strong>${alert.type.toUpperCase()}</strong>: ${alert.message}
                        <span style="float: right;">${new Date(alert.timestamp).toLocaleTimeString()}</span>
                    `;
                    container.insertBefore(alertDiv, container.firstChild);
                    
                    // Keep only last 10 alerts visible
                    while (container.children.length > 10) {
                        container.removeChild(container.lastChild);
                    }
                }
                
                // Initial data load
                fetch('/api/metrics/system')
                    .then(response => response.json())
                    .then(data => {
                        if (Object.keys(data).length > 0) {
                            updateMetrics(data);
                        }
                    });
                
                fetch('/api/alerts')
                    .then(response => response.json())
                    .then(alerts => {
                        alerts.forEach(alert => addAlert(alert));
                    });
            </script>
        </body>
        </html>
        """
    
    def record_request(self, processing_time: float, success: bool = True):
        """Record a request for metrics tracking."""
        self.total_requests += 1
        self.request_times.append(processing_time)
        
        if not success:
            self.error_count += 1
    
    def record_face_recognition(self, confidence: float, processing_time: float, 
                              faces_detected: int, success: bool = True):
        """Record face recognition metrics."""
        self.recognition_stats["total"] += 1
        
        if success:
            self.recognition_stats["successful"] += 1
            self.confidence_scores.append(confidence)
        else:
            self.recognition_stats["failed"] += 1
        
        self.recognition_stats["faces_detected"] += faces_detected
        self.processing_times.append(processing_time)
    
    def shutdown(self):
        """Shutdown dashboard monitoring."""
        self.monitoring_active = False

# Integration middleware for automatic metrics collection
class DashboardMiddleware:
    """Middleware to automatically collect metrics from requests."""
    
    def __init__(self, dashboard: SystemDashboard):
        self.dashboard = dashboard
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            try:
                # Process request
                await self.app(scope, receive, send)
                
                # Record successful request
                processing_time = time.time() - start_time
                self.dashboard.record_request(processing_time, success=True)
                
            except Exception as e:
                # Record failed request
                processing_time = time.time() - start_time
                self.dashboard.record_request(processing_time, success=False)
                raise
        else:
            await self.app(scope, receive, send)

# Example usage
if __name__ == "__main__":
    import uvicorn
    
    # Create dashboard
    dashboard = SystemDashboard()
    
    # Run dashboard server
    uvicorn.run(
        dashboard.app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )