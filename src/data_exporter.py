"""
Data export utilities for reporting and backups.

Supports CSV, JSON, Excel exports with streaming and compression.
"""

from typing import List, Dict, Any, Optional, AsyncIterator, Iterator, Union
from pathlib import Path
import csv
import json
import gzip
from datetime import datetime
from io import StringIO, BytesIO
import logging

logger = logging.getLogger(__name__)


class ExportFormat:
    """Export format constants."""
    
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"  # JSON Lines
    EXCEL = "excel"


class DataExporter:
    """Export data to various formats."""
    
    def __init__(
        self,
        compress: bool = False,
        chunk_size: int = 1000
    ):
        """
        Initialize data exporter.
        
        Args:
            compress: Enable gzip compression
            chunk_size: Chunk size for streaming
        """
        self.compress = compress
        self.chunk_size = chunk_size
    
    def export_to_csv(
        self,
        data: List[Dict[str, Any]],
        filepath: Union[str, Path],
        fieldnames: Optional[List[str]] = None
    ):
        """
        Export data to CSV file.
        
        Args:
            data: List of dictionaries
            filepath: Output file path
            fieldnames: CSV column names
        """
        if not data:
            logger.warning("No data to export")
            return
        
        # Auto-detect fieldnames
        if not fieldnames:
            fieldnames = list(data[0].keys())
        
        filepath = Path(filepath)
        
        # Open file with optional compression
        if self.compress:
            filepath = filepath.with_suffix(filepath.suffix + ".gz")
            file_obj = gzip.open(filepath, "wt", encoding="utf-8")
        else:
            file_obj = open(filepath, "w", encoding="utf-8", newline="")
        
        try:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
            
            logger.info(f"Exported {len(data)} rows to {filepath}")
        finally:
            file_obj.close()
    
    def export_to_json(
        self,
        data: Union[List[Dict[str, Any]], Dict[str, Any]],
        filepath: Union[str, Path],
        indent: Optional[int] = 2
    ):
        """
        Export data to JSON file.
        
        Args:
            data: Data to export
            filepath: Output file path
            indent: JSON indentation
        """
        filepath = Path(filepath)
        
        # Open file with optional compression
        if self.compress:
            filepath = filepath.with_suffix(filepath.suffix + ".gz")
            file_obj = gzip.open(filepath, "wt", encoding="utf-8")
        else:
            file_obj = open(filepath, "w", encoding="utf-8")
        
        try:
            json.dump(data, file_obj, indent=indent, default=str)
            
            count = len(data) if isinstance(data, list) else 1
            logger.info(f"Exported {count} records to {filepath}")
        finally:
            file_obj.close()
    
    def export_to_jsonl(
        self,
        data: List[Dict[str, Any]],
        filepath: Union[str, Path]
    ):
        """
        Export data to JSON Lines file.
        
        Args:
            data: List of dictionaries
            filepath: Output file path
        """
        filepath = Path(filepath)
        
        # Open file with optional compression
        if self.compress:
            filepath = filepath.with_suffix(filepath.suffix + ".gz")
            file_obj = gzip.open(filepath, "wt", encoding="utf-8")
        else:
            file_obj = open(filepath, "w", encoding="utf-8")
        
        try:
            for record in data:
                json.dump(record, file_obj, default=str)
                file_obj.write("\n")
            
            logger.info(f"Exported {len(data)} records to {filepath}")
        finally:
            file_obj.close()
    
    def stream_csv(
        self,
        data: Iterator[Dict[str, Any]],
        fieldnames: List[str]
    ) -> Iterator[str]:
        """
        Stream data as CSV.
        
        Args:
            data: Data iterator
            fieldnames: CSV column names
        
        Yields:
            CSV rows
        """
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        yield output.getvalue()
        output.truncate(0)
        output.seek(0)
        
        # Write rows
        for record in data:
            writer.writerow(record)
            yield output.getvalue()
            output.truncate(0)
            output.seek(0)
    
    def stream_jsonl(
        self,
        data: Iterator[Dict[str, Any]]
    ) -> Iterator[str]:
        """
        Stream data as JSON Lines.
        
        Args:
            data: Data iterator
        
        Yields:
            JSON Lines
        """
        for record in data:
            yield json.dumps(record, default=str) + "\n"
    
    async def stream_csv_async(
        self,
        data: AsyncIterator[Dict[str, Any]],
        fieldnames: List[str]
    ) -> AsyncIterator[str]:
        """
        Stream data as CSV (async).
        
        Args:
            data: Async data iterator
            fieldnames: CSV column names
        
        Yields:
            CSV rows
        """
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        yield output.getvalue()
        output.truncate(0)
        output.seek(0)
        
        # Write rows
        async for record in data:
            writer.writerow(record)
            yield output.getvalue()
            output.truncate(0)
            output.seek(0)
    
    async def stream_jsonl_async(
        self,
        data: AsyncIterator[Dict[str, Any]]
    ) -> AsyncIterator[str]:
        """
        Stream data as JSON Lines (async).
        
        Args:
            data: Async data iterator
        
        Yields:
            JSON Lines
        """
        async for record in data:
            yield json.dumps(record, default=str) + "\n"


class ReportGenerator:
    """Generate reports from data."""
    
    def __init__(self, exporter: Optional[DataExporter] = None):
        """Initialize report generator."""
        self.exporter = exporter or DataExporter()
    
    def generate_summary_report(
        self,
        data: List[Dict[str, Any]],
        group_by: str,
        metrics: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate summary report.
        
        Args:
            data: Input data
            group_by: Field to group by
            metrics: Fields to aggregate
        
        Returns:
            Summary data
        """
        summary: Dict[str, Dict[str, Any]] = {}
        
        for record in data:
            key = record.get(group_by)
            
            if key not in summary:
                summary[key] = {
                    group_by: key,
                    "count": 0
                }
                
                for metric in metrics:
                    summary[key][f"{metric}_sum"] = 0
                    summary[key][f"{metric}_min"] = float("inf")
                    summary[key][f"{metric}_max"] = float("-inf")
            
            summary[key]["count"] += 1
            
            for metric in metrics:
                value = record.get(metric, 0)
                
                summary[key][f"{metric}_sum"] += value
                summary[key][f"{metric}_min"] = min(
                    summary[key][f"{metric}_min"],
                    value
                )
                summary[key][f"{metric}_max"] = max(
                    summary[key][f"{metric}_max"],
                    value
                )
        
        # Calculate averages
        for key, stats in summary.items():
            for metric in metrics:
                stats[f"{metric}_avg"] = stats[f"{metric}_sum"] / stats["count"]
        
        return list(summary.values())
    
    def generate_time_series_report(
        self,
        data: List[Dict[str, Any]],
        timestamp_field: str,
        value_field: str,
        interval: str = "hour"
    ) -> List[Dict[str, Any]]:
        """
        Generate time series report.
        
        Args:
            data: Input data
            timestamp_field: Timestamp field name
            value_field: Value field name
            interval: Grouping interval (hour, day, week, month)
        
        Returns:
            Time series data
        """
        series: Dict[str, Dict[str, Any]] = {}
        
        for record in data:
            timestamp = record.get(timestamp_field)
            
            if not timestamp:
                continue
            
            # Parse timestamp
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            
            # Group by interval
            if interval == "hour":
                key = timestamp.strftime("%Y-%m-%d %H:00:00")
            elif interval == "day":
                key = timestamp.strftime("%Y-%m-%d")
            elif interval == "week":
                key = timestamp.strftime("%Y-W%W")
            elif interval == "month":
                key = timestamp.strftime("%Y-%m")
            else:
                key = str(timestamp)
            
            if key not in series:
                series[key] = {
                    "timestamp": key,
                    "count": 0,
                    "sum": 0
                }
            
            series[key]["count"] += 1
            series[key]["sum"] += record.get(value_field, 0)
        
        # Calculate averages
        for stats in series.values():
            stats["average"] = stats["sum"] / stats["count"] if stats["count"] > 0 else 0
        
        # Sort by timestamp
        result = sorted(series.values(), key=lambda x: x["timestamp"])
        
        return result
    
    def export_report(
        self,
        data: List[Dict[str, Any]],
        filepath: Union[str, Path],
        format: str = ExportFormat.CSV
    ):
        """
        Export report to file.
        
        Args:
            data: Report data
            filepath: Output file path
            format: Export format
        """
        if format == ExportFormat.CSV:
            self.exporter.export_to_csv(data, filepath)
        elif format == ExportFormat.JSON:
            self.exporter.export_to_json(data, filepath)
        elif format == ExportFormat.JSONL:
            self.exporter.export_to_jsonl(data, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Example usage:
"""
from src.data_exporter import DataExporter, ReportGenerator, ExportFormat

# Export data
exporter = DataExporter(compress=True)

data = [
    {"id": 1, "name": "Alice", "score": 95},
    {"id": 2, "name": "Bob", "score": 87}
]

exporter.export_to_csv(data, "output.csv")
exporter.export_to_json(data, "output.json")
exporter.export_to_jsonl(data, "output.jsonl")

# Generate reports
generator = ReportGenerator()

summary = generator.generate_summary_report(
    data=sales_data,
    group_by="region",
    metrics=["revenue", "quantity"]
)

time_series = generator.generate_time_series_report(
    data=events,
    timestamp_field="created_at",
    value_field="amount",
    interval="day"
)

generator.export_report(summary, "summary.csv", ExportFormat.CSV)
"""
