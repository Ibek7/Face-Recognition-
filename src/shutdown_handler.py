"""
Graceful shutdown handler for FastAPI applications.

Ensures clean shutdown of resources and ongoing operations.
"""

import asyncio
import signal
import sys
from typing import Callable, List
import logging

logger = logging.getLogger(__name__)


class ShutdownHandler:
    """Handle graceful application shutdown."""
    
    def __init__(self, timeout: int = 30):
        """
        Initialize shutdown handler.
        
        Args:
            timeout: Maximum time to wait for shutdown (seconds)
        """
        self.timeout = timeout
        self.shutdown_callbacks: List[Callable] = []
        self.is_shutting_down = False
    
    def register_callback(self, callback: Callable):
        """
        Register shutdown callback.
        
        Args:
            callback: Async function to call on shutdown
        """
        self.shutdown_callbacks.append(callback)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        # Handle SIGINT (Ctrl+C) and SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)
        
        logger.info("Shutdown signal handlers registered")
    
    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name
        logger.info(f"Received shutdown signal: {signal_name}")
        
        # Trigger shutdown
        asyncio.create_task(self.shutdown())
    
    async def shutdown(self):
        """Execute graceful shutdown."""
        if self.is_shutting_down:
            logger.warning("Shutdown already in progress")
            return
        
        self.is_shutting_down = True
        logger.info("Starting graceful shutdown...")
        
        try:
            # Execute all callbacks with timeout
            await asyncio.wait_for(
                self._execute_callbacks(),
                timeout=self.timeout
            )
            logger.info("Graceful shutdown completed")
        
        except asyncio.TimeoutError:
            logger.error(f"Shutdown timeout after {self.timeout}s")
        
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        finally:
            # Force exit
            sys.exit(0)
    
    async def _execute_callbacks(self):
        """Execute all shutdown callbacks."""
        tasks = []
        
        for callback in self.shutdown_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(callback())
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    tasks.append(loop.run_in_executor(None, callback))
            except Exception as e:
                logger.error(f"Error creating shutdown task: {e}")
        
        # Wait for all callbacks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Shutdown callback {i} failed: {result}"
                    )


# Global shutdown handler
shutdown_handler = ShutdownHandler()


async def cleanup_database_connections():
    """Example: Close database connections."""
    logger.info("Closing database connections...")
    await asyncio.sleep(0.1)  # Simulate cleanup
    logger.info("Database connections closed")


async def stop_background_tasks():
    """Example: Stop background tasks."""
    logger.info("Stopping background tasks...")
    await asyncio.sleep(0.1)  # Simulate cleanup
    logger.info("Background tasks stopped")


async def flush_caches():
    """Example: Flush caches."""
    logger.info("Flushing caches...")
    await asyncio.sleep(0.1)  # Simulate cleanup
    logger.info("Caches flushed")


# Example usage in api_server.py:
"""
from fastapi import FastAPI
from src.shutdown_handler import shutdown_handler

app = FastAPI()

@app.on_event("startup")
async def startup():
    # Register shutdown callbacks
    shutdown_handler.register_callback(cleanup_database_connections)
    shutdown_handler.register_callback(stop_background_tasks)
    shutdown_handler.register_callback(flush_caches)
    
    # Setup signal handlers
    shutdown_handler.setup_signal_handlers()

@app.on_event("shutdown")
async def shutdown():
    # FastAPI's native shutdown event
    await shutdown_handler.shutdown()

# Or manually trigger shutdown
# await shutdown_handler.shutdown()
"""
