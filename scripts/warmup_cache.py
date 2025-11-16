#!/usr/bin/env python3
"""
Cache Warmup Script

Pre-warms application caches on startup to improve initial performance:
- Model loading and compilation
- Embedding cache population
- Static data preloading
- Database connection pool initialization
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

import numpy as np
import redis.asyncio as aioredis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheWarmer:
    """Cache warmup orchestrator"""
    
    def __init__(
        self,
        database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/face_recognition",
        redis_url: str = "redis://localhost:6379/0",
        model_path: str = "models"
    ):
        self.database_url = database_url
        self.redis_url = redis_url
        self.model_path = Path(model_path)
        
        self.db_engine = None
        self.redis_client = None
        
        self.warmup_stats = {
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "tasks_completed": [],
            "tasks_failed": [],
            "cache_size_mb": 0
        }
    
    async def initialize(self):
        """Initialize connections"""
        logger.info("Initializing connections...")
        
        # Database
        try:
            self.db_engine = create_async_engine(
                self.database_url,
                pool_size=10,
                max_overflow=20,
                echo=False
            )
            logger.info("✓ Database engine created")
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
        
        # Redis
        try:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False
            )
            await self.redis_client.ping()
            logger.info("✓ Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    async def shutdown(self):
        """Cleanup connections"""
        if self.db_engine:
            await self.db_engine.dispose()
        
        if self.redis_client:
            await self.redis_client.close()
    
    async def warmup_all(self):
        """Execute all warmup tasks"""
        self.warmup_stats["start_time"] = time.time()
        
        logger.info("=" * 60)
        logger.info("Starting cache warmup...")
        logger.info("=" * 60)
        
        await self.initialize()
        
        # Run warmup tasks
        tasks = [
            self.warmup_models(),
            self.warmup_embeddings_cache(),
            self.warmup_person_cache(),
            self.warmup_config_cache(),
            self.warmup_database_pool(),
            self.warmup_static_data()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            task_name = tasks[i].__name__
            if isinstance(result, Exception):
                logger.error(f"✗ {task_name} failed: {result}")
                self.warmup_stats["tasks_failed"].append(task_name)
            else:
                logger.info(f"✓ {task_name} completed")
                self.warmup_stats["tasks_completed"].append(task_name)
        
        await self.shutdown()
        
        self.warmup_stats["end_time"] = time.time()
        self.warmup_stats["duration_seconds"] = (
            self.warmup_stats["end_time"] - self.warmup_stats["start_time"]
        )
        
        self._print_summary()
    
    async def warmup_models(self):
        """Load and compile ML models"""
        logger.info("Warming up ML models...")
        
        # Simulated model loading (replace with actual model loading)
        import torch
        
        models = {
            "detection": "yolov8n-face.pt",
            "recognition": "facenet512.pth"
        }
        
        for model_type, model_file in models.items():
            model_path = self.model_path / model_file
            
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue
            
            try:
                # Load model
                start_time = time.time()
                
                # Simulated loading
                await asyncio.sleep(0.5)
                
                # Run inference to compile (JIT)
                dummy_input = torch.randn(1, 3, 224, 224)
                
                # Simulated inference
                await asyncio.sleep(0.2)
                
                duration = time.time() - start_time
                logger.info(f"  ✓ {model_type} model loaded ({duration:.2f}s)")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to load {model_type} model: {e}")
    
    async def warmup_embeddings_cache(self):
        """Pre-load frequently accessed embeddings into Redis"""
        logger.info("Warming up embeddings cache...")
        
        if not self.redis_client or not self.db_engine:
            logger.warning("Redis or Database not available")
            return
        
        try:
            # Fetch all person embeddings from database
            async with AsyncSession(self.db_engine) as session:
                # Simulated query
                # result = await session.execute(
                #     select(Person.id, Person.name, Embedding.vector)
                #     .join(Embedding)
                # )
                
                # Simulated data
                persons = [
                    {"id": f"person_{i}", "name": f"Person {i}", "embedding": np.random.rand(512).tolist()}
                    for i in range(100)
                ]
                
                # Store in Redis
                cached_count = 0
                cache_size = 0
                
                for person in persons:
                    # Store embedding in Redis
                    key = f"embedding:{person['id']}"
                    value = json.dumps(person["embedding"])
                    
                    await self.redis_client.set(key, value, ex=3600)  # 1 hour TTL
                    
                    cached_count += 1
                    cache_size += len(value)
                
                logger.info(f"  ✓ Cached {cached_count} embeddings ({cache_size / 1024:.2f} KB)")
                self.warmup_stats["cache_size_mb"] += cache_size / (1024 * 1024)
                
        except Exception as e:
            logger.error(f"  ✗ Failed to warmup embeddings cache: {e}")
    
    async def warmup_person_cache(self):
        """Cache person metadata"""
        logger.info("Warming up person cache...")
        
        if not self.redis_client or not self.db_engine:
            logger.warning("Redis or Database not available")
            return
        
        try:
            # Simulated person data
            persons = [
                {
                    "id": f"person_{i}",
                    "name": f"Person {i}",
                    "metadata": {"department": "Engineering"}
                }
                for i in range(100)
            ]
            
            cached_count = 0
            cache_size = 0
            
            for person in persons:
                key = f"person:{person['id']}"
                value = json.dumps(person)
                
                await self.redis_client.set(key, value, ex=1800)  # 30 min TTL
                
                cached_count += 1
                cache_size += len(value)
            
            logger.info(f"  ✓ Cached {cached_count} persons ({cache_size / 1024:.2f} KB)")
            self.warmup_stats["cache_size_mb"] += cache_size / (1024 * 1024)
            
        except Exception as e:
            logger.error(f"  ✗ Failed to warmup person cache: {e}")
    
    async def warmup_config_cache(self):
        """Cache configuration data"""
        logger.info("Warming up config cache...")
        
        if not self.redis_client:
            logger.warning("Redis not available")
            return
        
        try:
            # Cache application configuration
            config_data = {
                "detection_model": "yolov8",
                "recognition_model": "facenet",
                "detection_confidence": 0.5,
                "recognition_confidence": 0.6,
                "max_faces": 10,
                "cache_ttl": 3600
            }
            
            for key, value in config_data.items():
                redis_key = f"config:{key}"
                await self.redis_client.set(redis_key, json.dumps(value), ex=86400)  # 24 hours
            
            logger.info(f"  ✓ Cached {len(config_data)} config values")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to warmup config cache: {e}")
    
    async def warmup_database_pool(self):
        """Initialize database connection pool"""
        logger.info("Warming up database connection pool...")
        
        if not self.db_engine:
            logger.warning("Database not available")
            return
        
        try:
            # Create initial connections
            connections = []
            
            for i in range(5):
                async with AsyncSession(self.db_engine) as session:
                    # Execute simple query to establish connection
                    await session.execute(select(1))
                    connections.append(session)
            
            logger.info(f"  ✓ Established {len(connections)} database connections")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to warmup database pool: {e}")
    
    async def warmup_static_data(self):
        """Load static data into memory"""
        logger.info("Warming up static data...")
        
        try:
            # Load reference data
            static_data = {
                "model_metadata": {
                    "yolov8": {"input_size": 640, "classes": 1},
                    "facenet": {"embedding_size": 512}
                },
                "supported_formats": ["jpeg", "png", "webp"],
                "max_image_size": 10 * 1024 * 1024,  # 10MB
                "rate_limits": {
                    "free": 100,
                    "premium": 1000
                }
            }
            
            if self.redis_client:
                # Cache static data
                for key, value in static_data.items():
                    redis_key = f"static:{key}"
                    await self.redis_client.set(
                        redis_key,
                        json.dumps(value),
                        ex=86400  # 24 hours
                    )
            
            logger.info(f"  ✓ Loaded {len(static_data)} static data items")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to warmup static data: {e}")
    
    def _print_summary(self):
        """Print warmup summary"""
        stats = self.warmup_stats
        
        logger.info("=" * 60)
        logger.info("Cache Warmup Summary")
        logger.info("=" * 60)
        logger.info(f"Duration: {stats['duration_seconds']:.2f} seconds")
        logger.info(f"Tasks completed: {len(stats['tasks_completed'])}")
        logger.info(f"Tasks failed: {len(stats['tasks_failed'])}")
        logger.info(f"Cache size: {stats['cache_size_mb']:.2f} MB")
        
        if stats['tasks_completed']:
            logger.info("\nCompleted tasks:")
            for task in stats['tasks_completed']:
                logger.info(f"  ✓ {task}")
        
        if stats['tasks_failed']:
            logger.info("\nFailed tasks:")
            for task in stats['tasks_failed']:
                logger.info(f"  ✗ {task}")
        
        logger.info("=" * 60)
    
    def save_stats(self, output_file: str = "cache_warmup_stats.json"):
        """Save warmup statistics to file"""
        with open(output_file, 'w') as f:
            json.dump(self.warmup_stats, f, indent=2)
        
        logger.info(f"Stats saved to {output_file}")


async def warmup_on_startup(
    database_url: Optional[str] = None,
    redis_url: Optional[str] = None,
    model_path: Optional[str] = None
):
    """
    Warmup function to be called on application startup
    
    Usage with FastAPI:
        @app.on_event("startup")
        async def startup_event():
            await warmup_on_startup()
    """
    warmer = CacheWarmer(
        database_url=database_url or "postgresql+asyncpg://postgres:postgres@localhost:5432/face_recognition",
        redis_url=redis_url or "redis://localhost:6379/0",
        model_path=model_path or "models"
    )
    
    await warmer.warmup_all()
    warmer.save_stats()


async def main():
    """Main entry point for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache warmup script")
    parser.add_argument(
        "--database-url",
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/face_recognition",
        help="Database URL"
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379/0",
        help="Redis URL"
    )
    parser.add_argument(
        "--model-path",
        default="models",
        help="Path to model directory"
    )
    parser.add_argument(
        "--stats-file",
        default="cache_warmup_stats.json",
        help="Output file for statistics"
    )
    
    args = parser.parse_args()
    
    warmer = CacheWarmer(
        database_url=args.database_url,
        redis_url=args.redis_url,
        model_path=args.model_path
    )
    
    await warmer.warmup_all()
    warmer.save_stats(args.stats_file)


if __name__ == "__main__":
    asyncio.run(main())
