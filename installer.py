# Path: codeSage.be/installer.py

import os
import sys
import logging
import subprocess
from pathlib import Path
from src.utils.security_manager import SecurityManager
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CodeSage-Installer")

class CodeSageInstaller:
    def __init__(self):
        self.root_dir = Path(__file__).parent.absolute()
        self.security_manager = SecurityManager()
        self.required_dirs = [
            'data/models',
            'data/cache',
            'data/model_cache',
            'logs',
            'config'
        ]

    def create_directories(self) -> bool:
        """Create required directories"""
        try:
            for dir_path in self.required_dirs:
                full_path = self.root_dir / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False

    def setup_python_environment(self) -> bool:
        """Install Python dependencies"""
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "-r", "requirements.txt"
            ])
            logger.info("Python dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Python dependencies: {e}")
            return False

    def setup_security(self) -> Optional[Dict[str, str]]:
        """Setup security and generate keys"""
        try:
            keys = self.security_manager.generate_installation_keys()
            self.security_manager.create_env_file()
            logger.info("Security setup completed")
            return keys
        except Exception as e:
            logger.error(f"Security setup failed: {e}")
            return None

    def initialize_database(self) -> bool:
        """Initialize SQLite database"""
        try:
            from src.models.database import init_db
            init_db()
            logger.info("Database initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False

    def verify_installation(self) -> bool:
        """Verify installation integrity"""
        checks = {
            "Directories": self.verify_directories(),
            "Security": self.security_manager.verify_installation(),
            "Database": self.verify_database(),
            "Environment": self.verify_environment()
        }
        
        failed_checks = [k for k, v in checks.items() if not v]
        if failed_checks:
            logger.error(f"Installation verification failed for: {', '.join(failed_checks)}")
            return False
            
        logger.info("Installation verified successfully")
        return True

    def verify_directories(self) -> bool:
        """Verify all required directories exist"""
        return all(
            (self.root_dir / dir_path).exists() 
            for dir_path in self.required_dirs
        )

    def verify_database(self) -> bool:
        """Verify database exists and is accessible"""
        try:
            from src.models.database import verify_db
            return verify_db()
        except:
            return False

    def verify_environment(self) -> bool:
        """Verify environment setup"""
        required_files = ['.env', 'config/security_config.enc']
        return all(
            (self.root_dir / file).exists() 
            for file in required_files
        )

def main():
    logger.info("Starting CodeSage Installation...")
    installer = CodeSageInstaller()

    steps = [
        ("Creating directories", installer.create_directories),
        ("Setting up Python environment", installer.setup_python_environment),
        ("Setting up security", installer.setup_security),
        ("Initializing database", installer.initialize_database),
        ("Verifying installation", installer.verify_installation)
    ]

    for step_name, step_func in steps:
        logger.info(f"Step: {step_name}")
        if not step_func():
            logger.error(f"Installation failed at: {step_name}")
            sys.exit(1)

    logger.info("CodeSage installation completed successfully!")
    logger.info("You can now start the application using: python src/app.py")

if __name__ == "__main__":
    main()