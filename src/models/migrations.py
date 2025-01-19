# Path: codeSage.be/src/models/migrations.py

import sqlite3
import os
import logging
from datetime import datetime
from ..config import active_config

logger = logging.getLogger(__name__)

MIGRATIONS = [
    # Migration 0001 - Initial Schema
    """
    CREATE TABLE IF NOT EXISTS licenses (
        id TEXT PRIMARY KEY,
        key TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        is_active BOOLEAN DEFAULT TRUE,
        metadata TEXT
    );
    
    CREATE TABLE IF NOT EXISTS model_cache (
        id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_used TIMESTAMP,
        size_bytes INTEGER,
        metadata TEXT
    );
    
    CREATE TABLE IF NOT EXISTS analysis_history (
        id TEXT PRIMARY KEY,
        project_path TEXT NOT NULL,
        analysis_type TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        status TEXT,
        results TEXT,
        metadata TEXT
    );
    
    CREATE TABLE IF NOT EXISTS codebase_metrics (
        id TEXT PRIMARY KEY,
        project_path TEXT NOT NULL,
        collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        total_files INTEGER,
        total_lines INTEGER,
        languages TEXT,
        complexity_score REAL,
        metrics_data TEXT
    );
    """,
    
    # Migration 0002 - Add User Settings
    """
    CREATE TABLE IF NOT EXISTS user_settings (
        id TEXT PRIMARY KEY,
        setting_key TEXT UNIQUE NOT NULL,
        setting_value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    INSERT OR IGNORE INTO user_settings (id, setting_key, setting_value)
    VALUES 
        ('default_model', 'model_name', 'Salesforce/codet5-base'),
        ('max_file_size', 'max_file_size_mb', '10'),
        ('cache_enabled', 'cache_enabled', 'true'),
        ('cache_ttl_days', 'cache_ttl_days', '7');
    """
]

def init_db():
    """Initialize database and run migrations"""
    db_path = active_config.DB_PATH
    
    # Create directory if doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create migrations table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                migration_name TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Get applied migrations
        cursor.execute("SELECT COUNT(*) FROM migrations")
        applied_count = cursor.fetchone()[0]
        
        # Apply pending migrations
        for i, migration in enumerate(MIGRATIONS[applied_count:], start=applied_count + 1):
            logger.info(f"Applying migration {i}...")
            cursor.executescript(migration)
            cursor.execute(
                "INSERT INTO migrations (migration_name) VALUES (?)",
                (f"migration_{i:04d}",)
            )
            conn.commit()
        
        logger.info("Database migrations completed successfully")
        return True
            
    except Exception as e:
        logger.error(f"Database migration failed: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def reset_db():
    """Reset database (for development only)"""
    if active_config.DEBUG:
        try:
            os.remove(active_config.DB_PATH)
            logger.info("Database reset successfully")
            init_db()
        except Exception as e:
            logger.error(f"Database reset failed: {str(e)}")
            raise
    else:
        raise RuntimeError("Database reset not allowed in production")

if __name__ == "__main__":
    init_db()