# Path: codeSage.be/src/tests/test_database.py

import pytest
import os
import json
from datetime import datetime, timedelta
from src.models.database import Database, DatabaseManager, init_db, verify_db

@pytest.fixture
def test_db():
    """Fixture to create test database"""
    test_db_path = './data/test_licenses.db'
    
    # Ensure test db directory exists
    os.makedirs(os.path.dirname(test_db_path), exist_ok=True)
    
    # Remove test db if exists
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        
    db = Database(test_db_path)
    init_db()  # Initialize test database
    yield db
    
    # Cleanup
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

@pytest.fixture
def db_manager():
    """Fixture for DatabaseManager"""
    return DatabaseManager()

def test_database_initialization(test_db):
    """Test database initialization"""
    assert verify_db()

def test_save_analysis(db_manager):
    """Test saving analysis record"""
    analysis_id = db_manager.save_analysis(
        project_path="/test/project",
        analysis_type="bug_detection",
        metadata={"test": "data"}
    )
    
    with db_manager.db as db:
        db.cursor.execute("SELECT * FROM analysis_history WHERE id=?", (analysis_id,))
        record = db.cursor.fetchone()
        
    assert record is not None
    assert record['project_path'] == "/test/project"
    assert record['analysis_type'] == "bug_detection"
    assert json.loads(record['metadata']) == {"test": "data"}

def test_update_analysis(db_manager):
    """Test updating analysis record"""
    # First create analysis
    analysis_id = db_manager.save_analysis(
        project_path="/test/project",
        analysis_type="code_review"
    )
    
    # Update it
    results = {"issues": ["issue1", "issue2"]}
    db_manager.update_analysis(analysis_id, "completed", results)
    
    # Verify update
    with db_manager.db as db:
        db.cursor.execute("SELECT * FROM analysis_history WHERE id=?", (analysis_id,))
        record = db.cursor.fetchone()
    
    assert record['status'] == "completed"
    assert json.loads(record['results']) == results
    assert record['completed_at'] is not None

def test_save_bug_report(db_manager):
    """Test saving bug report"""
    bug_id = db_manager.save_bug_report(
        file_path="/test/file.py",
        line_number=42,
        description="Test bug",
        severity="HIGH",
        metadata={"priority": "urgent"}
    )
    
    with db_manager.db as db:
        db.cursor.execute("SELECT * FROM bug_reports WHERE id=?", (bug_id,))
        record = db.cursor.fetchone()
    
    assert record is not None
    assert record['file_path'] == "/test/file.py"
    assert record['line_number'] == 42
    assert record['severity'] == "HIGH"
    assert record['status'] == "OPEN"
    assert json.loads(record['metadata']) == {"priority": "urgent"}

def test_license_validation(db_manager):
    """Test license validation"""
    # Insert test license
    test_key = "TEST-LICENSE-KEY"
    expires_at = (datetime.now() + timedelta(days=30)).isoformat()
    
    with db_manager.db as db:
        db.cursor.execute("""
            INSERT INTO licenses (id, key, expires_at, is_active)
            VALUES (?, ?, ?, TRUE)
        """, ('test_license', test_key, expires_at))
        db.conn.commit()
    
    # Test validation
    license_info = db_manager.validate_license(test_key)
    assert license_info is not None
    assert license_info['is_active'] is True

    # Test invalid key
    invalid_info = db_manager.validate_license("INVALID-KEY")
    assert invalid_info is None

def test_expired_license(db_manager):
    """Test expired license validation"""
    test_key = "EXPIRED-LICENSE"
    expires_at = (datetime.now() - timedelta(days=1)).isoformat()
    
    with db_manager.db as db:
        db.cursor.execute("""
            INSERT INTO licenses (id, key, expires_at, is_active)
            VALUES (?, ?, ?, TRUE)
        """, ('expired_license', test_key, expires_at))
        db.conn.commit()
    
    license_info = db_manager.validate_license(test_key)
    assert license_info is None

def test_database_connection(test_db):
    """Test database connection context manager"""
    with test_db as db:
        assert db.conn is not None
        assert db.cursor is not None
        
        # Test basic query
        db.cursor.execute("SELECT 1")
        result = db.cursor.fetchone()
        assert result is not None
    
    # Connection should be closed after context
    assert test_db.conn is None