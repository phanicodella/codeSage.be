# Path: codeSage.be/src/models/database.py

import sqlite3
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = './data/licenses.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cursor = None

    def connect(self):
        """Create database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

def init_db():
    """Initialize database with required tables"""
    with Database() as db:
        # Create licenses table
        db.cursor.execute("""
            CREATE TABLE IF NOT EXISTS licenses (
                id TEXT PRIMARY KEY,
                key TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                metadata TEXT
            )
        """)

        # Create analysis_history table
        db.cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id TEXT PRIMARY KEY,
                project_path TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT,
                results TEXT,
                metadata TEXT
            )
        """)

        # Create bug_reports table
        db.cursor.execute("""
            CREATE TABLE IF NOT EXISTS bug_reports (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                line_number INTEGER,
                description TEXT,
                severity TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                resolution TEXT,
                metadata TEXT
            )
        """)

        # Create codebase_metrics table
        db.cursor.execute("""
            CREATE TABLE IF NOT EXISTS codebase_metrics (
                id TEXT PRIMARY KEY,
                project_path TEXT NOT NULL,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_files INTEGER,
                total_lines INTEGER,
                languages TEXT,
                complexity_score REAL,
                metrics_data TEXT
            )
        """)

        db.conn.commit()

def verify_db() -> bool:
    """Verify database structure and accessibility"""
    try:
        with Database() as db:
            # Check if all required tables exist
            tables = [
                'licenses',
                'analysis_history',
                'bug_reports',
                'codebase_metrics'
            ]
            
            for table in tables:
                db.cursor.execute(f"""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (table,))
                
                if not db.cursor.fetchone():
                    logger.error(f"Required table missing: {table}")
                    return False

            # Test write access
            test_id = 'test_' + datetime.now().strftime('%Y%m%d%H%M%S')
            db.cursor.execute("""
                INSERT INTO codebase_metrics (id, project_path, total_files)
                VALUES (?, ?, ?)
            """, (test_id, '/test/path', 0))
            
            db.cursor.execute("DELETE FROM codebase_metrics WHERE id=?", (test_id,))
            db.conn.commit()
            
            return True
            
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False

class DatabaseManager:
    """Manager class for database operations"""
    
    def __init__(self):
        self.db = Database()

    def save_analysis(self, project_path: str, analysis_type: str, metadata: Dict = None) -> str:
        """Save new analysis record"""
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        with self.db as db:
            db.cursor.execute("""
                INSERT INTO analysis_history (id, project_path, analysis_type, metadata)
                VALUES (?, ?, ?, ?)
            """, (analysis_id, project_path, analysis_type, json.dumps(metadata)))
            db.conn.commit()
            
        return analysis_id

    def update_analysis(self, analysis_id: str, status: str, results: Dict = None):
        """Update existing analysis record"""
        with self.db as db:
            db.cursor.execute("""
                UPDATE analysis_history 
                SET status=?, results=?, completed_at=CURRENT_TIMESTAMP
                WHERE id=?
            """, (status, json.dumps(results) if results else None, analysis_id))
            db.conn.commit()

    def save_bug_report(self, file_path: str, line_number: int, description: str, 
                       severity: str, metadata: Dict = None) -> str:
        """Save new bug report"""
        bug_id = f"bug_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        with self.db as db:
            db.cursor.execute("""
                INSERT INTO bug_reports (
                    id, file_path, line_number, description, severity, 
                    status, metadata
                )
                VALUES (?, ?, ?, ?, ?, 'OPEN', ?)
            """, (bug_id, file_path, line_number, description, severity, 
                  json.dumps(metadata)))
            db.conn.commit()
            
        return bug_id

    def validate_license(self, license_key: str) -> Optional[Dict[str, Any]]:
        """Validate license key"""
        with self.db as db:
            db.cursor.execute("""
                SELECT id, created_at, expires_at, is_active, metadata
                FROM licenses WHERE key=? AND is_active=TRUE 
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """, (license_key,))
            
            row = db.cursor.fetchone()
            if row:
                return dict(row)
            return None

if __name__ == "__main__":
    init_db()
    if verify_db():
        print("Database initialized successfully")
    else:
        print("Database initialization failed")