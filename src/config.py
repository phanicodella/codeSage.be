# Path: codeSage.be/src/config.py

import os
from pathlib import Path
from typing import Dict, Any
import json
import logging
from logging.handlers import RotatingFileHandler
import secrets
from datetime import timedelta

class Config:
    """Base configuration for CodeSage"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent.absolute()
    DATA_DIR = BASE_DIR / 'data'
    CACHE_DIR = DATA_DIR / 'cache'
    MODEL_DIR = DATA_DIR / 'models'
    LOG_DIR = BASE_DIR / 'logs'
    UPLOAD_DIR = BASE_DIR / 'uploads'
    
    # Application settings
    VERSION = "1.0.0"
    APP_NAME = "CodeSage"
    DEBUG = False
    TESTING = False
    
    # API settings
    API_HOST = os.getenv("CODESAGE_API_HOST", "localhost")
    API_PORT = int(os.getenv("CODESAGE_API_PORT", "5000"))
    SECRET_KEY = os.getenv("CODESAGE_SECRET_KEY", secrets.token_hex(32))
    
    # CORS settings
    CORS_ORIGINS = ["http://localhost:3000"]  # Frontend URL
    
    # JWT settings
    JWT_SECRET_KEY = os.getenv("CODESAGE_JWT_SECRET", secrets.token_hex(32))
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=30)
    
    # Database settings
    DB_PATH = DATA_DIR / 'licenses.db'
    
    # Model settings
    DEFAULT_MODEL = "Salesforce/codet5-base"
    MODEL_CACHE_ENABLED = True
    MODEL_MAX_LENGTH = 512
    MODEL_TEMPERATURE = 0.7
    MODEL_TOP_P = 0.95
    
    # File processing settings
    ALLOWED_EXTENSIONS = {
        '.py', '.js', '.jsx', '.ts', '.tsx',
        '.java', '.cpp', '.hpp', '.c', '.h',
        '.cs', '.rb', '.php', '.go', '.rs'
    }
    
    # Cache settings
    CACHE_ENABLED = True
    CACHE_TTL = timedelta(days=7)
    
    # WebSocket settings
    WS_HOST = os.getenv("CODESAGE_WS_HOST", "localhost")
    WS_PORT = int(os.getenv("CODESAGE_WS_PORT", "8765"))
    WS_PING_INTERVAL = int(os.getenv("CODESAGE_WS_PING_INTERVAL", "30"))
    WS_PING_TIMEOUT = int(os.getenv("CODESAGE_WS_PING_TIMEOUT", "10"))

    # Visualization settings
    VIS_UPDATE_INTERVAL = int(os.getenv("CODESAGE_VIS_UPDATE_INTERVAL", "1000"))
    VIS_MAX_NODES = int(os.getenv("CODESAGE_VIS_MAX_NODES", "1000"))
    VIS_MAX_EDGES = int(os.getenv("CODESAGE_VIS_MAX_EDGES", "5000"))

    # Logging settings
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = LOG_DIR / 'codesage.log'
    LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5

    @classmethod
    def initialize(cls):
        """Initialize application directories and settings"""
        # Create necessary directories
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        cls.setup_logging()
        
        # Initialize database
        cls.init_database()

    @classmethod
    def setup_logging(cls):
        """Configure application logging"""
        logging.basicConfig(
            level=cls.LOG_LEVEL,
            format=cls.LOG_FORMAT,
            handlers=[
                RotatingFileHandler(
                    cls.LOG_FILE,
                    maxBytes=cls.LOG_MAX_SIZE,
                    backupCount=cls.LOG_BACKUP_COUNT
                ),
                logging.StreamHandler()
            ]
        )

    @classmethod
    def init_database(cls):
        """Initialize SQLite database"""
        import sqlite3
        
        cls.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(cls.DB_PATH)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS licenses (
                id TEXT PRIMARY KEY,
                key TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id TEXT PRIMARY KEY,
                project_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_count INTEGER,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for API consumption"""
        return {
            "version": self.VERSION,
            "model": {
                "name": self.DEFAULT_MODEL,
                "max_length": self.MODEL_MAX_LENGTH,
                "temperature": self.MODEL_TEMPERATURE
            },
            "files": {
                "allowed_extensions": list(self.ALLOWED_EXTENSIONS)
            },
            "cache": {
                "enabled": self.CACHE_ENABLED,
                "ttl_days": self.CACHE_TTL.days
            }
        }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    CORS_ORIGINS = ["*"]
    LOG_LEVEL = logging.DEBUG

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    DB_PATH = ':memory:'  # Use in-memory database
    CACHE_ENABLED = False

class ProductionConfig(Config):
    """Production configuration"""
    LOG_LEVEL = logging.WARNING
    CORS_ORIGINS = []  # Configure based on deployment
    MODEL_CACHE_ENABLED = True
    CACHE_ENABLED = True

# Configuration mapping
config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig
}

# Active configuration
active_config = config_by_name[os.getenv("CODESAGE_ENV", "development")]

# Initialize configuration
active_config.initialize()