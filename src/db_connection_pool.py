# Database Connection Pooling System

import sqlite3
import threading
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from queue import Queue, Empty
from contextlib import contextmanager
import logging

@dataclass
class PoolStatistics:
    """Statistics for connection pool."""
    total_connections: int
    active_connections: int
    idle_connections: int
    wait_time_avg_ms: float
    created_count: int
    closed_count: int
    failed_requests: int
    timestamp: float

class PooledConnection:
    """Wrapper for pooled database connection."""
    
    def __init__(self, conn_id: str, connection: sqlite3.Connection):
        self.conn_id = conn_id
        self.connection = connection
        self.created_at = time.time()
        self.last_used = self.created_at
        self.is_active = False
        self.lifetime_queries = 0
    
    def mark_active(self):
        """Mark connection as active."""
        self.is_active = True
        self.last_used = time.time()
    
    def mark_idle(self):
        """Mark connection as idle."""
        self.is_active = False
        self.last_used = time.time()
    
    def get_age_seconds(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at
    
    def get_idle_time_seconds(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_used

class ConnectionPool:
    """Database connection pool for efficient resource management."""
    
    def __init__(self, database: str, min_connections: int = 5, 
                 max_connections: int = 20, max_lifetime_seconds: int = 3600):
        self.database = database
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_lifetime_seconds = max_lifetime_seconds
        
        self.available_connections: Queue = Queue(maxsize=max_connections)
        self.all_connections: Dict[str, PooledConnection] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        self.statistics = {
            'created_count': 0,
            'closed_count': 0,
            'failed_requests': 0,
            'wait_times': []
        }
        
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool with minimum connections."""
        for _ in range(self.min_connections):
            try:
                conn = self._create_connection()
                self.available_connections.put(conn, block=False)
            except Exception as e:
                self.logger.error(f"Failed to initialize pool: {e}")
    
    def _create_connection(self) -> PooledConnection:
        """Create new database connection."""
        with self.lock:
            conn_id = f"conn_{len(self.all_connections)}"
            
            try:
                sqlite_conn = sqlite3.connect(self.database, check_same_thread=False)
                pooled_conn = PooledConnection(conn_id, sqlite_conn)
                
                self.all_connections[conn_id] = pooled_conn
                self.statistics['created_count'] += 1
                
                self.logger.debug(f"Created connection: {conn_id}")
                
                return pooled_conn
            
            except Exception as e:
                self.logger.error(f"Failed to create connection: {e}")
                raise
    
    @contextmanager
    def get_connection(self, timeout_seconds: float = 5.0):
        """Get connection from pool."""
        
        start_time = time.time()
        conn = None
        
        try:
            # Try to get available connection
            try:
                conn = self.available_connections.get(timeout=timeout_seconds)
            except Empty:
                # Create new connection if pool not full
                with self.lock:
                    if len(self.all_connections) < self.max_connections:
                        conn = self._create_connection()
                    else:
                        self.statistics['failed_requests'] += 1
                        raise RuntimeError("Connection pool exhausted")
            
            # Check if connection is still valid
            if not self._is_connection_valid(conn):
                conn = self._create_connection()
            
            conn.mark_active()
            
            # Record wait time
            wait_time_ms = (time.time() - start_time) * 1000
            self.statistics['wait_times'].append(wait_time_ms)
            
            yield conn
        
        finally:
            if conn:
                conn.mark_idle()
                
                # Return to pool if still valid
                if self._is_connection_valid(conn):
                    try:
                        self.available_connections.put(conn, block=False)
                    except Exception as e:
                        self.logger.warning(f"Failed to return connection: {e}")
                        self._close_connection(conn)
                else:
                    self._close_connection(conn)
    
    def _is_connection_valid(self, conn: PooledConnection) -> bool:
        """Check if connection is valid."""
        
        # Check if connection exceeds max lifetime
        if conn.get_age_seconds() > self.max_lifetime_seconds:
            return False
        
        # Try to ping connection
        try:
            conn.connection.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def _close_connection(self, conn: PooledConnection):
        """Close and remove connection from pool."""
        try:
            conn.connection.close()
            
            with self.lock:
                if conn.conn_id in self.all_connections:
                    del self.all_connections[conn.conn_id]
                self.statistics['closed_count'] += 1
            
            self.logger.debug(f"Closed connection: {conn.conn_id}")
        
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute query using pooled connection."""
        
        with self.get_connection() as conn:
            cursor = conn.connection.cursor()
            cursor.execute(query, params)
            
            # Fetch results
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            results = [dict(zip(columns, row)) for row in rows]
            
            conn.connection.commit()
            cursor.close()
            
            return results
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute update query using pooled connection."""
        
        with self.get_connection() as conn:
            cursor = conn.connection.cursor()
            cursor.execute(query, params)
            
            affected_rows = cursor.rowcount
            
            conn.connection.commit()
            cursor.close()
            
            return affected_rows
    
    def close_all(self):
        """Close all connections in pool."""
        with self.lock:
            for conn in list(self.all_connections.values()):
                try:
                    conn.connection.close()
                except Exception as e:
                    self.logger.error(f"Error closing connection: {e}")
            
            self.all_connections.clear()
            
            # Clear queue
            while not self.available_connections.empty():
                try:
                    self.available_connections.get_nowait()
                except Empty:
                    break
    
    def get_statistics(self) -> PoolStatistics:
        """Get pool statistics."""
        with self.lock:
            active_connections = sum(
                1 for conn in self.all_connections.values() if conn.is_active
            )
            idle_connections = len(self.all_connections) - active_connections
            
            wait_times = self.statistics['wait_times']
            avg_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
            
            return PoolStatistics(
                total_connections=len(self.all_connections),
                active_connections=active_connections,
                idle_connections=idle_connections,
                wait_time_avg_ms=avg_wait_time,
                created_count=self.statistics['created_count'],
                closed_count=self.statistics['closed_count'],
                failed_requests=self.statistics['failed_requests'],
                timestamp=time.time()
            )
    
    def maintenance(self):
        """Perform pool maintenance (cleanup idle connections)."""
        with self.lock:
            connections_to_remove = []
            
            for conn_id, conn in self.all_connections.items():
                # Remove idle connections exceeding max lifetime
                if (not conn.is_active and 
                    conn.get_age_seconds() > self.max_lifetime_seconds):
                    connections_to_remove.append(conn_id)
            
            for conn_id in connections_to_remove:
                try:
                    self.all_connections[conn_id].connection.close()
                    del self.all_connections[conn_id]
                    self.statistics['closed_count'] += 1
                except Exception as e:
                    self.logger.error(f"Error during maintenance: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_all()

# Global pool instance
_connection_pool: Optional[ConnectionPool] = None

def get_connection_pool(database: str = "face_recognition.db", 
                       min_connections: int = 5,
                       max_connections: int = 20) -> ConnectionPool:
    """Get global connection pool."""
    global _connection_pool
    
    if _connection_pool is None:
        _connection_pool = ConnectionPool(
            database=database,
            min_connections=min_connections,
            max_connections=max_connections
        )
    
    return _connection_pool

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create pool
    pool = get_connection_pool()
    
    # Execute queries
    try:
        # Create table
        pool.execute_update(
            "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)"
        )
        
        # Insert data
        pool.execute_update(
            "INSERT INTO users (name) VALUES (?)",
            ("John Doe",)
        )
        
        # Query data
        results = pool.execute_query("SELECT * FROM users")
        print(f"Users: {results}")
        
        # Get statistics
        stats = pool.get_statistics()
        print(f"\nPool Statistics:")
        print(f"  Total connections: {stats.total_connections}")
        print(f"  Active connections: {stats.active_connections}")
        print(f"  Idle connections: {stats.idle_connections}")
        print(f"  Avg wait time: {stats.wait_time_avg_ms:.2f}ms")
        print(f"  Created: {stats.created_count}, Closed: {stats.closed_count}")
    
    finally:
        pool.close_all()
