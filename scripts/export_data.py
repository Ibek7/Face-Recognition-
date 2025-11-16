#!/usr/bin/env python3
"""
Data Export Utility

Export face recognition data in various formats:
- JSON (standard format)
- CSV (tabular data)
- Excel (XLSX with multiple sheets)
- XML (structured export)
- SQL (database dumps)
- Parquet (efficient columnar format)
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum
import io

from pydantic import BaseModel
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    XML = "xml"
    SQL = "sql"
    PARQUET = "parquet"


class ExportOptions(BaseModel):
    """Export configuration options"""
    include_embeddings: bool = True
    include_metadata: bool = True
    include_timestamps: bool = True
    pretty_print: bool = True
    compression: Optional[str] = None  # 'gzip', 'bz2', 'zip'


class PersonData(BaseModel):
    """Person data model for export"""
    id: str
    name: str
    email: Optional[str] = None
    metadata: Dict[str, Any] = {}
    embedding_count: int = 0
    embeddings: List[List[float]] = []
    created_at: datetime
    updated_at: datetime


class DataExporter:
    """Export face recognition data in various formats"""
    
    def __init__(self, output_dir: str = "exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(
        self,
        data: List[PersonData],
        format: ExportFormat,
        filename: Optional[str] = None,
        options: Optional[ExportOptions] = None
    ) -> Path:
        """
        Export data to specified format
        
        Args:
            data: List of person data to export
            format: Export format
            filename: Output filename (auto-generated if None)
            options: Export options
        
        Returns:
            Path to exported file
        """
        options = options or ExportOptions()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_recognition_export_{timestamp}.{format.value}"
        
        output_path = self.output_dir / filename
        
        logger.info(f"Exporting {len(data)} records to {format.value} format...")
        
        if format == ExportFormat.JSON:
            self._export_json(data, output_path, options)
        
        elif format == ExportFormat.CSV:
            self._export_csv(data, output_path, options)
        
        elif format == ExportFormat.EXCEL:
            self._export_excel(data, output_path, options)
        
        elif format == ExportFormat.XML:
            self._export_xml(data, output_path, options)
        
        elif format == ExportFormat.SQL:
            self._export_sql(data, output_path, options)
        
        elif format == ExportFormat.PARQUET:
            self._export_parquet(data, output_path, options)
        
        logger.info(f"✓ Export complete: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
        
        return output_path
    
    def _export_json(
        self,
        data: List[PersonData],
        output_path: Path,
        options: ExportOptions
    ):
        """Export to JSON format"""
        export_data = []
        
        for person in data:
            person_dict = {
                "id": person.id,
                "name": person.name,
                "email": person.email
            }
            
            if options.include_metadata:
                person_dict["metadata"] = person.metadata
            
            if options.include_timestamps:
                person_dict["created_at"] = person.created_at.isoformat()
                person_dict["updated_at"] = person.updated_at.isoformat()
            
            if options.include_embeddings:
                person_dict["embedding_count"] = person.embedding_count
                person_dict["embeddings"] = person.embeddings
            
            export_data.append(person_dict)
        
        # Write JSON
        with open(output_path, 'w') as f:
            if options.pretty_print:
                json.dump(export_data, f, indent=2, default=str)
            else:
                json.dump(export_data, f, default=str)
    
    def _export_csv(
        self,
        data: List[PersonData],
        output_path: Path,
        options: ExportOptions
    ):
        """Export to CSV format"""
        # Flatten data for CSV
        rows = []
        
        for person in data:
            row = {
                "id": person.id,
                "name": person.name,
                "email": person.email or "",
                "embedding_count": person.embedding_count
            }
            
            if options.include_timestamps:
                row["created_at"] = person.created_at.isoformat()
                row["updated_at"] = person.updated_at.isoformat()
            
            if options.include_metadata:
                # Flatten metadata
                for key, value in person.metadata.items():
                    row[f"metadata_{key}"] = value
            
            # Note: Embeddings are too large for CSV, stored as count only
            
            rows.append(row)
        
        # Write CSV
        if rows:
            fieldnames = rows[0].keys()
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    
    def _export_excel(
        self,
        data: List[PersonData],
        output_path: Path,
        options: ExportOptions
    ):
        """Export to Excel format with multiple sheets"""
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Persons
            persons_data = []
            for person in data:
                person_row = {
                    "ID": person.id,
                    "Name": person.name,
                    "Email": person.email or "",
                    "Embedding Count": person.embedding_count
                }
                
                if options.include_timestamps:
                    person_row["Created At"] = person.created_at
                    person_row["Updated At"] = person.updated_at
                
                persons_data.append(person_row)
            
            df_persons = pd.DataFrame(persons_data)
            df_persons.to_excel(writer, sheet_name='Persons', index=False)
            
            # Sheet 2: Metadata (if included)
            if options.include_metadata:
                metadata_rows = []
                for person in data:
                    for key, value in person.metadata.items():
                        metadata_rows.append({
                            "Person ID": person.id,
                            "Key": key,
                            "Value": str(value)
                        })
                
                if metadata_rows:
                    df_metadata = pd.DataFrame(metadata_rows)
                    df_metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Sheet 3: Summary
            summary_data = {
                "Metric": [
                    "Total Persons",
                    "Total Embeddings",
                    "Avg Embeddings per Person",
                    "Export Date"
                ],
                "Value": [
                    len(data),
                    sum(p.embedding_count for p in data),
                    round(sum(p.embedding_count for p in data) / len(data), 2) if data else 0,
                    datetime.now().isoformat()
                ]
            }
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    def _export_xml(
        self,
        data: List[PersonData],
        output_path: Path,
        options: ExportOptions
    ):
        """Export to XML format"""
        from xml.etree.ElementTree import Element, SubElement, ElementTree
        from xml.dom import minidom
        
        # Create root element
        root = Element('face_recognition_export')
        root.set('version', '1.0')
        root.set('export_date', datetime.now().isoformat())
        root.set('record_count', str(len(data)))
        
        # Add persons
        persons_elem = SubElement(root, 'persons')
        
        for person in data:
            person_elem = SubElement(persons_elem, 'person')
            person_elem.set('id', person.id)
            
            # Basic fields
            name_elem = SubElement(person_elem, 'name')
            name_elem.text = person.name
            
            if person.email:
                email_elem = SubElement(person_elem, 'email')
                email_elem.text = person.email
            
            # Embedding count
            embedding_count_elem = SubElement(person_elem, 'embedding_count')
            embedding_count_elem.text = str(person.embedding_count)
            
            # Timestamps
            if options.include_timestamps:
                created_elem = SubElement(person_elem, 'created_at')
                created_elem.text = person.created_at.isoformat()
                
                updated_elem = SubElement(person_elem, 'updated_at')
                updated_elem.text = person.updated_at.isoformat()
            
            # Metadata
            if options.include_metadata and person.metadata:
                metadata_elem = SubElement(person_elem, 'metadata')
                for key, value in person.metadata.items():
                    meta_elem = SubElement(metadata_elem, 'item')
                    meta_elem.set('key', key)
                    meta_elem.text = str(value)
        
        # Pretty print XML
        xml_str = minidom.parseString(
            ElementTree.tostring(root, encoding='utf-8')
        ).toprettyxml(indent="  ")
        
        with open(output_path, 'w') as f:
            f.write(xml_str)
    
    def _export_sql(
        self,
        data: List[PersonData],
        output_path: Path,
        options: ExportOptions
    ):
        """Export to SQL insert statements"""
        sql_statements = []
        
        # Create table statement
        create_table = """
CREATE TABLE IF NOT EXISTS persons (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    embedding_count INTEGER,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
"""
        sql_statements.append(create_table)
        sql_statements.append("")
        
        # Insert statements
        for person in data:
            escaped_name = person.name.replace("'", "''")  # Escape single quotes
            values = [
                f"'{person.id}'",
                f"'{escaped_name}'",
                f"'{person.email}'" if person.email else "NULL",
                str(person.embedding_count),
                f"'{person.created_at.isoformat()}'" if options.include_timestamps else "NULL",
                f"'{person.updated_at.isoformat()}'" if options.include_timestamps else "NULL"
            ]
            
            insert_stmt = f"INSERT INTO persons VALUES ({', '.join(values)});"
            sql_statements.append(insert_stmt)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(sql_statements))
    
    def _export_parquet(
        self,
        data: List[PersonData],
        output_path: Path,
        options: ExportOptions
    ):
        """Export to Parquet format (efficient columnar storage)"""
        # Prepare data for DataFrame
        records = []
        
        for person in data:
            record = {
                "id": person.id,
                "name": person.name,
                "email": person.email,
                "embedding_count": person.embedding_count
            }
            
            if options.include_timestamps:
                record["created_at"] = person.created_at
                record["updated_at"] = person.updated_at
            
            if options.include_metadata:
                # Flatten metadata
                for key, value in person.metadata.items():
                    record[f"metadata_{key}"] = value
            
            records.append(record)
        
        # Create DataFrame and export
        df = pd.DataFrame(records)
        
        compression = options.compression or 'snappy'
        df.to_parquet(output_path, compression=compression, index=False)
    
    def export_summary(self, data: List[PersonData]) -> Dict[str, Any]:
        """Generate export summary statistics"""
        if not data:
            return {
                "total_persons": 0,
                "total_embeddings": 0,
                "avg_embeddings_per_person": 0
            }
        
        total_embeddings = sum(p.embedding_count for p in data)
        
        return {
            "total_persons": len(data),
            "total_embeddings": total_embeddings,
            "avg_embeddings_per_person": round(total_embeddings / len(data), 2),
            "persons_with_email": sum(1 for p in data if p.email),
            "earliest_created": min(p.created_at for p in data).isoformat(),
            "latest_created": max(p.created_at for p in data).isoformat()
        }


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = [
        PersonData(
            id=f"person_{i}",
            name=f"Person {i}",
            email=f"person{i}@example.com" if i % 2 == 0 else None,
            metadata={"department": "Engineering", "role": "Developer"},
            embedding_count=5,
            embeddings=[[0.1, 0.2, 0.3] for _ in range(5)],
            created_at=datetime(2024, 1, i+1),
            updated_at=datetime(2024, 1, i+1)
        )
        for i in range(10)
    ]
    
    # Create exporter
    exporter = DataExporter(output_dir="exports")
    
    # Export options
    options = ExportOptions(
        include_embeddings=False,  # Exclude for smaller files
        include_metadata=True,
        include_timestamps=True,
        pretty_print=True
    )
    
    print("Exporting data in various formats...\n")
    
    # Export to all formats
    for format in ExportFormat:
        try:
            output_file = exporter.export(sample_data, format, options=options)
            print(f"✓ {format.value.upper()}: {output_file}")
        except Exception as e:
            print(f"✗ {format.value.upper()}: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    
    summary = exporter.export_summary(sample_data)
    for key, value in summary.items():
        print(f"{key}: {value}")
