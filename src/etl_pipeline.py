# Data Pipeline & ETL Framework

import threading
from typing import List, Callable, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

class StageType(Enum):
    """Pipeline stage types."""
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    VALIDATE = "validate"
    FILTER = "filter"

@dataclass
class PipelineStage:
    """ETL pipeline stage."""
    name: str
    stage_type: StageType
    processor: Callable
    error_handling: str = "skip"  # skip, fail, retry
    max_retries: int = 3

class ETLPipeline:
    """Extract-Transform-Load pipeline."""
    
    def __init__(self, name: str):
        self.name = name
        self.stages: List[PipelineStage] = []
        self.stats = {'processed': 0, 'failed': 0, 'skipped': 0}
        self.lock = threading.RLock()
    
    def add_stage(self, stage: PipelineStage) -> 'ETLPipeline':
        """Add pipeline stage."""
        with self.lock:
            self.stages.append(stage)
        return self
    
    def execute(self, data: List[Any]) -> List[Any]:
        """Execute pipeline."""
        with self.lock:
            current_data = data
            
            for stage in self.stages:
                current_data = self._execute_stage(stage, current_data)
            
            return current_data
    
    def _execute_stage(self, stage: PipelineStage, data: List[Any]) -> List[Any]:
        """Execute single stage."""
        results = []
        
        for item in data:
            try:
                result = stage.processor(item)
                if result is not None:
                    results.append(result)
                else:
                    self.stats['skipped'] += 1
            
            except Exception as e:
                self.stats['failed'] += 1
                if stage.error_handling == 'fail':
                    raise
        
        self.stats['processed'] += len(data)
        return results
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        with self.lock:
            return self.stats.copy()

class DataExtractor:
    """Extract data from sources."""
    
    def __init__(self, source_type: str):
        self.source_type = source_type
        self.data = []
    
    def extract(self, source: str, **kwargs) -> List[Any]:
        """Extract data."""
        if self.source_type == "csv":
            return self._extract_csv(source)
        elif self.source_type == "json":
            return self._extract_json(source)
        elif self.source_type == "database":
            return self._extract_database(source, **kwargs)
        return []
    
    def _extract_csv(self, filepath: str) -> List[Dict]:
        """Extract from CSV."""
        import csv
        data = []
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)
        except Exception as e:
            print(f"Error extracting CSV: {e}")
        return data
    
    def _extract_json(self, filepath: str) -> List[Dict]:
        """Extract from JSON."""
        import json
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        except Exception as e:
            print(f"Error extracting JSON: {e}")
        return []
    
    def _extract_database(self, query: str, **kwargs) -> List[Dict]:
        """Extract from database."""
        # Mock implementation
        return []

class DataTransformer:
    """Transform data."""
    
    @staticmethod
    def map_fields(data: Dict, mapping: Dict[str, str]) -> Dict:
        """Map field names."""
        return {v: data.get(k) for k, v in mapping.items()}
    
    @staticmethod
    def filter_fields(data: Dict, fields: List[str]) -> Dict:
        """Filter fields."""
        return {k: v for k, v in data.items() if k in fields}
    
    @staticmethod
    def aggregate(data: List[Dict], key: str) -> Dict[str, List]:
        """Aggregate data by key."""
        result = {}
        for item in data:
            k = item.get(key)
            if k not in result:
                result[k] = []
            result[k].append(item)
        return result
    
    @staticmethod
    def enrich(data: Dict, enrichment_data: Dict) -> Dict:
        """Enrich data."""
        return {**data, **enrichment_data}
    
    @staticmethod
    def apply_function(data: Dict, func: Callable) -> Dict:
        """Apply custom function."""
        return func(data)

class DataValidator:
    """Validate data quality."""
    
    def __init__(self):
        self.rules: List[Callable] = []
    
    def add_rule(self, rule: Callable) -> None:
        """Add validation rule."""
        self.rules.append(rule)
    
    def validate(self, data: Any) -> bool:
        """Validate data."""
        for rule in self.rules:
            if not rule(data):
                return False
        return True
    
    def validate_batch(self, data: List[Any]) -> Dict:
        """Validate batch."""
        valid = []
        invalid = []
        
        for item in data:
            if self.validate(item):
                valid.append(item)
            else:
                invalid.append(item)
        
        return {
            'valid': valid,
            'invalid': invalid,
            'valid_count': len(valid),
            'invalid_count': len(invalid)
        }

class DataLoader:
    """Load data to targets."""
    
    def __init__(self, target_type: str):
        self.target_type = target_type
    
    def load(self, data: List[Any], target: str, **kwargs) -> Dict:
        """Load data."""
        if self.target_type == "csv":
            return self._load_csv(data, target)
        elif self.target_type == "json":
            return self._load_json(data, target)
        elif self.target_type == "database":
            return self._load_database(data, target, **kwargs)
        return {'loaded': 0, 'failed': 0}
    
    def _load_csv(self, data: List[Dict], filepath: str) -> Dict:
        """Load to CSV."""
        import csv
        try:
            if not data:
                return {'loaded': 0, 'failed': 0}
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            
            return {'loaded': len(data), 'failed': 0}
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return {'loaded': 0, 'failed': len(data)}
    
    def _load_json(self, data: List[Dict], filepath: str) -> Dict:
        """Load to JSON."""
        import json
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return {'loaded': len(data), 'failed': 0}
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return {'loaded': 0, 'failed': len(data)}
    
    def _load_database(self, data: List[Dict], table: str, **kwargs) -> Dict:
        """Load to database."""
        return {'loaded': len(data), 'failed': 0}

class PipelineBuilder:
    """Builder for ETL pipelines."""
    
    def __init__(self, name: str):
        self.pipeline = ETLPipeline(name)
    
    def extract(self, source_type: str, source: str) -> 'PipelineBuilder':
        """Add extract stage."""
        extractor = DataExtractor(source_type)
        
        def extract_func(_):
            return extractor.extract(source)
        
        self.pipeline.add_stage(PipelineStage(
            name=f"extract_{source_type}",
            stage_type=StageType.EXTRACT,
            processor=extract_func
        ))
        return self
    
    def transform(self, transformer_func: Callable) -> 'PipelineBuilder':
        """Add transform stage."""
        self.pipeline.add_stage(PipelineStage(
            name="transform",
            stage_type=StageType.TRANSFORM,
            processor=transformer_func
        ))
        return self
    
    def validate(self, validator_func: Callable) -> 'PipelineBuilder':
        """Add validation stage."""
        self.pipeline.add_stage(PipelineStage(
            name="validate",
            stage_type=StageType.VALIDATE,
            processor=validator_func
        ))
        return self
    
    def load(self, target_type: str, target: str) -> 'PipelineBuilder':
        """Add load stage."""
        loader = DataLoader(target_type)
        
        def load_func(data):
            return loader.load([data], target)
        
        self.pipeline.add_stage(PipelineStage(
            name=f"load_{target_type}",
            stage_type=StageType.LOAD,
            processor=load_func
        ))
        return self
    
    def build(self) -> ETLPipeline:
        """Build pipeline."""
        return self.pipeline

# Example usage
if __name__ == "__main__":
    # Create transformer
    def transform_func(item):
        if isinstance(item, dict):
            return {**item, 'processed': True}
        return item
    
    # Create validator
    def validate_func(item):
        return isinstance(item, dict) and 'id' in item
    
    # Build pipeline
    pipeline = PipelineBuilder("sample_etl") \
        .transform(transform_func) \
        .build()
    
    # Execute
    test_data = [{'id': 1}, {'id': 2}]
    results = pipeline.execute(test_data)
    
    print(f"Processed: {len(results)} items")
    print(f"Stats: {pipeline.get_stats()}")
