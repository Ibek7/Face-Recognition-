# Advanced Analytics and Reporting System for Face Recognition

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import io
import base64
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import DBSCAN
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AnalyticsConfig:
    """Configuration for analytics system."""
    database_path: str = "face_recognition_analytics.db"
    export_format: str = "html"  # html, pdf, json, csv
    time_aggregation: str = "hour"  # minute, hour, day, week, month
    confidence_threshold: float = 0.7
    include_images: bool = False
    anonymize_data: bool = True
    generate_heatmaps: bool = True
    enable_clustering: bool = True

@dataclass
class AnalyticsMetrics:
    """Comprehensive analytics metrics."""
    total_detections: int
    unique_persons: int
    recognition_accuracy: float
    avg_processing_time: float
    peak_detection_time: str
    most_recognized_person: str
    detection_rate_by_hour: Dict[int, int]
    confidence_distribution: List[float]
    false_positive_rate: float
    false_negative_rate: float
    system_uptime: float
    error_rate: float

class FaceRecognitionAnalytics:
    """Advanced analytics system for face recognition data."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.db_path = config.database_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self.init_database()
        
        # Color schemes for visualizations
        self.color_schemes = {
            'primary': ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'],
            'gradient': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'],
            'professional': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
    
    def init_database(self):
        """Initialize SQLite database for analytics storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                person_id TEXT,
                confidence REAL,
                processing_time REAL,
                frame_id TEXT,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_width INTEGER,
                bbox_height INTEGER,
                is_known_person BOOLEAN,
                session_id TEXT,
                camera_source TEXT,
                image_path TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                person_id TEXT PRIMARY KEY,
                name TEXT,
                first_seen DATETIME,
                last_seen DATETIME,
                total_detections INTEGER DEFAULT 0,
                avg_confidence REAL,
                notes TEXT,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                fps REAL,
                cpu_usage REAL,
                memory_usage REAL,
                gpu_usage REAL,
                queue_size INTEGER,
                error_count INTEGER,
                uptime_seconds REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT,
                generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                time_range_start DATETIME,
                time_range_end DATETIME,
                report_data TEXT,
                file_path TEXT,
                config_used TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info("Analytics database initialized")
    
    def log_detection(self, detection_data: Dict[str, Any]):
        """Log a face detection event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (
                person_id, confidence, processing_time, frame_id,
                bbox_x, bbox_y, bbox_width, bbox_height,
                is_known_person, session_id, camera_source, image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            detection_data.get('person_id'),
            detection_data.get('confidence', 0.0),
            detection_data.get('processing_time', 0.0),
            detection_data.get('frame_id'),
            detection_data.get('bbox', [0, 0, 0, 0])[0],
            detection_data.get('bbox', [0, 0, 0, 0])[1],
            detection_data.get('bbox', [0, 0, 0, 0])[2],
            detection_data.get('bbox', [0, 0, 0, 0])[3],
            detection_data.get('is_known_person', False),
            detection_data.get('session_id'),
            detection_data.get('camera_source'),
            detection_data.get('image_path')
        ))
        
        # Update person statistics
        if detection_data.get('person_id'):
            self.update_person_stats(detection_data['person_id'], detection_data.get('confidence', 0.0))
        
        conn.commit()
        conn.close()
    
    def update_person_stats(self, person_id: str, confidence: float):
        """Update person detection statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if person exists
        cursor.execute('SELECT * FROM persons WHERE person_id = ?', (person_id,))
        person = cursor.fetchone()
        
        if person:
            # Update existing person
            new_total = person[4] + 1
            new_avg_confidence = (person[5] * person[4] + confidence) / new_total
            
            cursor.execute('''
                UPDATE persons SET 
                    last_seen = CURRENT_TIMESTAMP,
                    total_detections = ?,
                    avg_confidence = ?
                WHERE person_id = ?
            ''', (new_total, new_avg_confidence, person_id))
        else:
            # Insert new person
            cursor.execute('''
                INSERT INTO persons (
                    person_id, first_seen, last_seen, total_detections, avg_confidence
                ) VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1, ?)
            ''', (person_id, confidence))
        
        conn.commit()
        conn.close()
    
    def log_system_metrics(self, metrics: Dict[str, Any]):
        """Log system performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_metrics (
                fps, cpu_usage, memory_usage, gpu_usage, 
                queue_size, error_count, uptime_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.get('fps', 0.0),
            metrics.get('cpu_usage', 0.0),
            metrics.get('memory_usage', 0.0),
            metrics.get('gpu_usage', 0.0),
            metrics.get('queue_size', 0),
            metrics.get('error_count', 0),
            metrics.get('uptime_seconds', 0.0)
        ))
        
        conn.commit()
        conn.close()
    
    def get_analytics_metrics(self, start_time: Optional[datetime] = None, 
                            end_time: Optional[datetime] = None) -> AnalyticsMetrics:
        """Calculate comprehensive analytics metrics."""
        
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        
        # Basic detection metrics
        detections_df = pd.read_sql_query('''
            SELECT * FROM detections 
            WHERE timestamp BETWEEN ? AND ?
        ''', conn, params=(start_time, end_time))
        
        persons_df = pd.read_sql_query('SELECT * FROM persons', conn)
        
        # Calculate metrics
        total_detections = len(detections_df)
        unique_persons = len(detections_df['person_id'].dropna().unique())
        
        # Recognition accuracy (known vs unknown)
        known_detections = len(detections_df[detections_df['is_known_person'] == True])
        recognition_accuracy = (known_detections / total_detections * 100) if total_detections > 0 else 0
        
        # Processing performance
        avg_processing_time = detections_df['processing_time'].mean() if len(detections_df) > 0 else 0
        
        # Peak detection time
        if len(detections_df) > 0:
            detections_df['hour'] = pd.to_datetime(detections_df['timestamp']).dt.hour
            hourly_counts = detections_df.groupby('hour').size()
            peak_hour = hourly_counts.idxmax()
            peak_detection_time = f"{peak_hour}:00"
            detection_rate_by_hour = hourly_counts.to_dict()
        else:
            peak_detection_time = "N/A"
            detection_rate_by_hour = {}
        
        # Most recognized person
        if len(detections_df) > 0 and not detections_df['person_id'].dropna().empty:
            person_counts = detections_df['person_id'].value_counts()
            most_recognized_person = person_counts.index[0] if len(person_counts) > 0 else "N/A"
        else:
            most_recognized_person = "N/A"
        
        # Confidence distribution
        confidence_distribution = detections_df['confidence'].dropna().tolist()
        
        # Error rates (simplified calculation)
        false_positive_rate = 0.05  # Would need ground truth data
        false_negative_rate = 0.03  # Would need ground truth data
        
        # System metrics
        system_df = pd.read_sql_query('''
            SELECT * FROM system_metrics 
            WHERE timestamp BETWEEN ? AND ?
        ''', conn, params=(start_time, end_time))
        
        system_uptime = system_df['uptime_seconds'].max() if len(system_df) > 0 else 0
        error_rate = system_df['error_count'].sum() / len(system_df) if len(system_df) > 0 else 0
        
        conn.close()
        
        return AnalyticsMetrics(
            total_detections=total_detections,
            unique_persons=unique_persons,
            recognition_accuracy=recognition_accuracy,
            avg_processing_time=avg_processing_time,
            peak_detection_time=peak_detection_time,
            most_recognized_person=most_recognized_person,
            detection_rate_by_hour=detection_rate_by_hour,
            confidence_distribution=confidence_distribution,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            system_uptime=system_uptime,
            error_rate=error_rate
        )
    
    def generate_comprehensive_report(self, start_time: Optional[datetime] = None, 
                                   end_time: Optional[datetime] = None,
                                   output_path: str = "analytics_report.html") -> str:
        """Generate a comprehensive analytics report."""
        
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()
        
        # Get data
        metrics = self.get_analytics_metrics(start_time, end_time)
        
        # Create visualizations
        charts = self.create_comprehensive_charts(start_time, end_time)
        
        # Generate HTML report
        html_content = self.generate_html_report(metrics, charts, start_time, end_time)
        
        # Save report
        output_file = Path(output_path)
        output_file.write_text(html_content, encoding='utf-8')
        
        # Save to database
        self.save_report_to_db("comprehensive", start_time, end_time, html_content, str(output_file))
        
        self.logger.info(f"Comprehensive report generated: {output_file}")
        return str(output_file)
    
    def create_comprehensive_charts(self, start_time: datetime, end_time: datetime) -> Dict[str, str]:
        """Create comprehensive charts for the report."""
        
        conn = sqlite3.connect(self.db_path)
        
        # Get data
        detections_df = pd.read_sql_query('''
            SELECT * FROM detections 
            WHERE timestamp BETWEEN ? AND ?
        ''', conn, params=(start_time, end_time))
        
        system_df = pd.read_sql_query('''
            SELECT * FROM system_metrics 
            WHERE timestamp BETWEEN ? AND ?
        ''', conn, params=(start_time, end_time))
        
        persons_df = pd.read_sql_query('SELECT * FROM persons', conn)
        
        charts = {}
        
        # 1. Detection Timeline
        if len(detections_df) > 0:
            detections_df['timestamp'] = pd.to_datetime(detections_df['timestamp'])
            hourly_detections = detections_df.set_index('timestamp').resample('H').size()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_detections.index,
                y=hourly_detections.values,
                mode='lines+markers',
                name='Detections per Hour',
                line=dict(color='#3498db', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title='Face Detection Timeline',
                xaxis_title='Time',
                yaxis_title='Number of Detections',
                template='plotly_white',
                height=400
            )
            
            charts['timeline'] = fig.to_html(include_plotlyjs='inline', div_id="timeline_chart")
        
        # 2. Confidence Distribution
        if len(detections_df) > 0 and 'confidence' in detections_df.columns:
            fig = px.histogram(
                detections_df, 
                x='confidence', 
                nbins=20,
                title='Confidence Score Distribution',
                color_discrete_sequence=['#2ecc71']
            )
            fig.update_layout(template='plotly_white', height=400)
            charts['confidence'] = fig.to_html(include_plotlyjs='inline', div_id="confidence_chart")
        
        # 3. Person Recognition Frequency
        if len(detections_df) > 0 and 'person_id' in detections_df.columns:
            person_counts = detections_df['person_id'].value_counts().head(10)
            
            fig = px.bar(
                x=person_counts.index,
                y=person_counts.values,
                title='Top 10 Most Recognized Persons',
                color=person_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                xaxis_title='Person ID',
                yaxis_title='Detection Count',
                template='plotly_white',
                height=400
            )
            charts['persons'] = fig.to_html(include_plotlyjs='inline', div_id="persons_chart")
        
        # 4. System Performance Over Time
        if len(system_df) > 0:
            system_df['timestamp'] = pd.to_datetime(system_df['timestamp'])
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('FPS', 'CPU Usage (%)', 'Memory Usage (%)', 'Queue Size'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # FPS
            fig.add_trace(
                go.Scatter(x=system_df['timestamp'], y=system_df['fps'], name='FPS'),
                row=1, col=1
            )
            
            # CPU Usage
            fig.add_trace(
                go.Scatter(x=system_df['timestamp'], y=system_df['cpu_usage'], name='CPU'),
                row=1, col=2
            )
            
            # Memory Usage
            fig.add_trace(
                go.Scatter(x=system_df['timestamp'], y=system_df['memory_usage'], name='Memory'),
                row=2, col=1
            )
            
            # Queue Size
            fig.add_trace(
                go.Scatter(x=system_df['timestamp'], y=system_df['queue_size'], name='Queue'),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="System Performance Metrics",
                template='plotly_white',
                height=600,
                showlegend=False
            )
            
            charts['performance'] = fig.to_html(include_plotlyjs='inline', div_id="performance_chart")
        
        # 5. Detection Heatmap (by hour and day of week)
        if len(detections_df) > 0:
            detections_df['hour'] = pd.to_datetime(detections_df['timestamp']).dt.hour
            detections_df['day_of_week'] = pd.to_datetime(detections_df['timestamp']).dt.day_name()
            
            heatmap_data = detections_df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(day_order)
            
            fig = px.imshow(
                heatmap_data,
                title='Detection Heatmap (Day of Week vs Hour)',
                color_continuous_scale='Viridis',
                aspect='auto'
            )
            fig.update_layout(
                xaxis_title='Hour of Day',
                yaxis_title='Day of Week',
                template='plotly_white',
                height=400
            )
            
            charts['heatmap'] = fig.to_html(include_plotlyjs='inline', div_id="heatmap_chart")
        
        # 6. Processing Time Analysis
        if len(detections_df) > 0 and 'processing_time' in detections_df.columns:
            fig = px.box(
                detections_df,
                y='processing_time',
                title='Processing Time Distribution',
                color_discrete_sequence=['#e74c3c']
            )
            fig.update_layout(template='plotly_white', height=400)
            charts['processing_time'] = fig.to_html(include_plotlyjs='inline', div_id="processing_chart")
        
        conn.close()
        return charts
    
    def generate_html_report(self, metrics: AnalyticsMetrics, charts: Dict[str, str], 
                           start_time: datetime, end_time: datetime) -> str:
        """Generate HTML report with metrics and charts."""
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Face Recognition Analytics Report</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    color: #2c3e50;
                }}
                .header {{
                    background: linear-gradient(135deg, #3498db, #2ecc71);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 1.2em;
                    opacity: 0.9;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 2.5em;
                    font-weight: bold;
                    color: #3498db;
                    margin-bottom: 10px;
                }}
                .metric-label {{
                    font-size: 1.1em;
                    color: #7f8c8d;
                }}
                .chart-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    margin-bottom: 30px;
                }}
                .summary-section {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    margin-bottom: 30px;
                }}
                .summary-section h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .insights {{
                    background: #ecf0f1;
                    padding: 20px;
                    border-radius: 8px;
                    margin-top: 20px;
                }}
                .insights h3 {{
                    color: #27ae60;
                    margin-top: 0;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Face Recognition Analytics Report</h1>
                <p>Analysis Period: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}</p>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{metrics.total_detections:,}</div>
                    <div class="metric-label">Total Detections</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.unique_persons}</div>
                    <div class="metric-label">Unique Persons</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.recognition_accuracy:.1f}%</div>
                    <div class="metric-label">Recognition Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.avg_processing_time:.2f}s</div>
                    <div class="metric-label">Avg Processing Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.peak_detection_time}</div>
                    <div class="metric-label">Peak Detection Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.system_uptime/3600:.1f}h</div>
                    <div class="metric-label">System Uptime</div>
                </div>
            </div>
            
            <div class="summary-section">
                <h2>📊 Executive Summary</h2>
                <p>
                    During the analysis period, the face recognition system processed <strong>{metrics.total_detections:,} face detections</strong> 
                    across <strong>{metrics.unique_persons} unique individuals</strong>. The system achieved an overall recognition accuracy 
                    of <strong>{metrics.recognition_accuracy:.1f}%</strong> with an average processing time of 
                    <strong>{metrics.avg_processing_time:.2f} seconds</strong> per detection.
                </p>
                
                <div class="insights">
                    <h3>🔍 Key Insights</h3>
                    <ul>
                        <li>Peak detection activity occurs at <strong>{metrics.peak_detection_time}</strong></li>
                        <li>Most frequently recognized person: <strong>{metrics.most_recognized_person}</strong></li>
                        <li>System maintained <strong>{metrics.system_uptime/3600:.1f} hours</strong> of uptime</li>
                        <li>Error rate: <strong>{metrics.error_rate:.2%}</strong></li>
                        <li>Average confidence score: <strong>{np.mean(metrics.confidence_distribution):.2f}</strong></li>
                    </ul>
                </div>
            </div>
        """
        
        # Add charts
        for chart_name, chart_html in charts.items():
            html_template += f'''
            <div class="chart-container">
                {chart_html}
            </div>
            '''
        
        # Add footer
        html_template += f"""
            <div class="footer">
                <p>Report generated by Face Recognition Analytics System v2.0</p>
                <p>For questions or support, please contact the system administrator.</p>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def generate_performance_report(self, start_time: Optional[datetime] = None, 
                                  end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate detailed performance analysis report."""
        
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        
        # System metrics analysis
        system_df = pd.read_sql_query('''
            SELECT * FROM system_metrics 
            WHERE timestamp BETWEEN ? AND ?
        ''', conn, params=(start_time, end_time))
        
        detections_df = pd.read_sql_query('''
            SELECT * FROM detections 
            WHERE timestamp BETWEEN ? AND ?
        ''', conn, params=(start_time, end_time))
        
        performance_metrics = {}
        
        if len(system_df) > 0:
            performance_metrics.update({
                'avg_fps': system_df['fps'].mean(),
                'min_fps': system_df['fps'].min(),
                'max_fps': system_df['fps'].max(),
                'fps_std': system_df['fps'].std(),
                'avg_cpu_usage': system_df['cpu_usage'].mean(),
                'max_cpu_usage': system_df['cpu_usage'].max(),
                'avg_memory_usage': system_df['memory_usage'].mean(),
                'max_memory_usage': system_df['memory_usage'].max(),
                'avg_queue_size': system_df['queue_size'].mean(),
                'max_queue_size': system_df['queue_size'].max(),
            })
        
        if len(detections_df) > 0:
            performance_metrics.update({
                'avg_processing_time': detections_df['processing_time'].mean(),
                'min_processing_time': detections_df['processing_time'].min(),
                'max_processing_time': detections_df['processing_time'].max(),
                'processing_time_p95': detections_df['processing_time'].quantile(0.95),
                'processing_time_p99': detections_df['processing_time'].quantile(0.99),
                'detections_per_minute': len(detections_df) / ((end_time - start_time).total_seconds() / 60),
            })
        
        # Performance scoring
        performance_score = self.calculate_performance_score(performance_metrics)
        performance_metrics['overall_score'] = performance_score
        
        conn.close()
        return performance_metrics
    
    def calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score (0-100)."""
        score = 100.0
        
        # FPS penalty
        if 'avg_fps' in metrics:
            if metrics['avg_fps'] < 15:
                score -= 30
            elif metrics['avg_fps'] < 25:
                score -= 15
        
        # CPU usage penalty
        if 'avg_cpu_usage' in metrics:
            if metrics['avg_cpu_usage'] > 80:
                score -= 20
            elif metrics['avg_cpu_usage'] > 60:
                score -= 10
        
        # Memory usage penalty
        if 'avg_memory_usage' in metrics:
            if metrics['avg_memory_usage'] > 80:
                score -= 15
            elif metrics['avg_memory_usage'] > 60:
                score -= 8
        
        # Processing time penalty
        if 'avg_processing_time' in metrics:
            if metrics['avg_processing_time'] > 1.0:
                score -= 25
            elif metrics['avg_processing_time'] > 0.5:
                score -= 12
        
        return max(0, score)
    
    def generate_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Generate trend analysis for the specified number of days."""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        
        # Daily detection counts
        detections_df = pd.read_sql_query('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM detections 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', conn, params=(start_time, end_time))
        
        # Calculate trends
        trends = {}
        
        if len(detections_df) > 1:
            # Linear regression for detection trend
            x = np.arange(len(detections_df))
            y = detections_df['count'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            trends['detection_trend'] = {
                'slope': slope,
                'direction': 'increasing' if slope > 0 else 'decreasing',
                'correlation': r_value,
                'significance': 'significant' if p_value < 0.05 else 'not_significant'
            }
        
        # Weekly patterns
        detections_weekly = pd.read_sql_query('''
            SELECT strftime('%w', timestamp) as day_of_week, 
                   AVG(CAST(strftime('%H', timestamp) AS INTEGER)) as avg_hour,
                   COUNT(*) as count
            FROM detections 
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY strftime('%w', timestamp)
        ''', conn, params=(start_time, end_time))
        
        trends['weekly_patterns'] = detections_weekly.to_dict('records')
        
        conn.close()
        return trends
    
    def export_raw_data(self, start_time: Optional[datetime] = None, 
                       end_time: Optional[datetime] = None,
                       format: str = 'csv') -> str:
        """Export raw data in specified format."""
        
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        
        # Get all relevant data
        detections_df = pd.read_sql_query('''
            SELECT * FROM detections 
            WHERE timestamp BETWEEN ? AND ?
        ''', conn, params=(start_time, end_time))
        
        system_df = pd.read_sql_query('''
            SELECT * FROM system_metrics 
            WHERE timestamp BETWEEN ? AND ?
        ''', conn, params=(start_time, end_time))
        
        persons_df = pd.read_sql_query('SELECT * FROM persons', conn)
        
        # Export based on format
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            detections_file = f'detections_{timestamp_str}.csv'
            system_file = f'system_metrics_{timestamp_str}.csv'
            persons_file = f'persons_{timestamp_str}.csv'
            
            detections_df.to_csv(detections_file, index=False)
            system_df.to_csv(system_file, index=False)
            persons_df.to_csv(persons_file, index=False)
            
            return f"Data exported to {detections_file}, {system_file}, {persons_file}"
        
        elif format == 'json':
            export_data = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'total_detections': len(detections_df),
                    'total_persons': len(persons_df)
                },
                'detections': detections_df.to_dict('records'),
                'system_metrics': system_df.to_dict('records'),
                'persons': persons_df.to_dict('records')
            }
            
            export_file = f'face_recognition_data_{timestamp_str}.json'
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return f"Data exported to {export_file}"
        
        conn.close()
    
    def save_report_to_db(self, report_type: str, start_time: datetime, 
                         end_time: datetime, report_data: str, file_path: str):
        """Save generated report to database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analytics_reports (
                report_type, time_range_start, time_range_end, 
                report_data, file_path, config_used
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            report_type, start_time, end_time, 
            report_data, file_path, json.dumps(asdict(self.config))
        ))
        
        conn.commit()
        conn.close()
    
    def cleanup_old_data(self, retention_days: int = 90):
        """Clean up old data based on retention policy."""
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete old detections
        cursor.execute('DELETE FROM detections WHERE timestamp < ?', (cutoff_date,))
        detections_deleted = cursor.rowcount
        
        # Delete old system metrics
        cursor.execute('DELETE FROM system_metrics WHERE timestamp < ?', (cutoff_date,))
        metrics_deleted = cursor.rowcount
        
        # Delete old reports
        cursor.execute('DELETE FROM analytics_reports WHERE generated_at < ?', (cutoff_date,))
        reports_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Cleanup completed: {detections_deleted} detections, {metrics_deleted} metrics, {reports_deleted} reports deleted")
        
        return {
            'detections_deleted': detections_deleted,
            'metrics_deleted': metrics_deleted,
            'reports_deleted': reports_deleted
        }


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = AnalyticsConfig(
        database_path="test_analytics.db",
        export_format="html",
        time_aggregation="hour",
        confidence_threshold=0.7,
        include_images=False,
        anonymize_data=True
    )
    
    # Initialize analytics system
    analytics = FaceRecognitionAnalytics(config)
    
    # Generate sample data
    print("Generating sample data...")
    import random
    
    # Sample detections
    for i in range(1000):
        detection_data = {
            'person_id': f"person_{random.randint(1, 20)}",
            'confidence': random.uniform(0.5, 0.99),
            'processing_time': random.uniform(0.1, 2.0),
            'frame_id': f"frame_{i}",
            'bbox': [random.randint(50, 200), random.randint(50, 200), 
                    random.randint(80, 150), random.randint(80, 150)],
            'is_known_person': random.choice([True, False]),
            'session_id': f"session_{random.randint(1, 5)}",
            'camera_source': 'webcam_0'
        }
        analytics.log_detection(detection_data)
    
    # Sample system metrics
    for i in range(100):
        metrics_data = {
            'fps': random.uniform(20, 35),
            'cpu_usage': random.uniform(30, 80),
            'memory_usage': random.uniform(40, 70),
            'gpu_usage': random.uniform(20, 60),
            'queue_size': random.randint(0, 10),
            'error_count': random.randint(0, 3),
            'uptime_seconds': i * 60
        }
        analytics.log_system_metrics(metrics_data)
    
    print("Sample data generated!")
    
    # Generate comprehensive report
    print("Generating comprehensive report...")
    report_path = analytics.generate_comprehensive_report()
    print(f"Report generated: {report_path}")
    
    # Generate performance report
    print("Generating performance report...")
    performance_metrics = analytics.generate_performance_report()
    print(f"Performance Score: {performance_metrics.get('overall_score', 0):.1f}/100")
    
    # Generate trend analysis
    print("Generating trend analysis...")
    trends = analytics.generate_trend_analysis(days=7)
    print(f"Detection trend: {trends.get('detection_trend', {}).get('direction', 'unknown')}")
    
    # Export data
    print("Exporting raw data...")
    export_result = analytics.export_raw_data(format='json')
    print(f"Export result: {export_result}")
    
    print("Analytics system demonstration completed!")