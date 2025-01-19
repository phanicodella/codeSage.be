# Path: codeSage.be/src/models/model_manager.py

import os
from pathlib import Path
import json
import hashlib
import shutil
import tempfile
import logging
from typing import Dict, Optional, List, Any
import torch
from transformers import AutoModelForSeq2SeqGeneration, AutoTokenizer
from huggingface_hub import snapshot_download
import requests
from tqdm import tqdm
from datetime import datetime
from config import active_config as config

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages offline models including downloading, caching, and versioning.
    Ensures models are available for offline use and handles updates when online.
    """

    def __init__(self):
        self.models_dir = config.MODEL_DIR
        self.cache_dir = config.CACHE_DIR
        self.model_info_file = self.models_dir / 'model_info.json'
        self.models_info = self._load_models_info()
        
        # Create necessary directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_models_info(self) -> Dict[str, Any]:
        """Load model information from JSON file"""
        if self.model_info_file.exists():
            try:
                with open(self.model_info_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading model info: {e}")
                return {}
        return {}

    def _save_models_info(self):
        """Save model information to JSON file"""
        try:
            with open(self.model_info_file, 'w') as f:
                json.dump(self.models_info, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model info: {e}")

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get local path for a model"""
        if model_name in self.models_info:
            return self.models_dir / model_name
        return None

    async def ensure_model_available(self, model_name: str, force_download: bool = False) -> Path:
        """
        Ensure model is available locally, downloading if necessary.
        
        Args:
            model_name: Name of the model (e.g., "Salesforce/codet5-base")
            force_download: Force download even if model exists locally
            
        Returns:
            Path to local model directory
        """
        model_path = self.get_model_path(model_name)
        
        if not force_download and model_path and model_path.exists():
            logger.info(f"Model {model_name} already available locally")
            return model_path

        logger.info(f"Downloading model {model_name}")
        try:
            # Download model files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download model using huggingface_hub
                snapshot_path = snapshot_download(
                    repo_id=model_name,
                    cache_dir=temp_dir,
                    local_files_only=False
                )

                # Calculate model hash
                model_hash = self._calculate_directory_hash(snapshot_path)
                
                # Create model directory
                model_path = self.models_dir / model_name
                model_path.mkdir(parents=True, exist_ok=True)
                
                # Copy files to model directory
                shutil.copytree(snapshot_path, model_path, dirs_exist_ok=True)
                
                # Update model info
                self.models_info[model_name] = {
                    'hash': model_hash,
                    'download_date': datetime.utcnow().isoformat(),
                    'last_verified': datetime.utcnow().isoformat(),
                    'size': self._get_directory_size(model_path)
                }
                
                self._save_models_info()
                logger.info(f"Model {model_name} downloaded successfully")
                
                return model_path
        
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            raise

    def verify_model_integrity(self, model_name: str) -> bool:
        """Verify model files integrity"""
        model_path = self.get_model_path(model_name)
        if not model_path or not model_path.exists():
            return False

        try:
            current_hash = self._calculate_directory_hash(model_path)
            stored_hash = self.models_info.get(model_name, {}).get('hash')
            
            if stored_hash and current_hash == stored_hash:
                # Update last verification date
                self.models_info[model_name]['last_verified'] = datetime.utcnow().isoformat()
                self._save_models_info()
                return True
                
            return False
        
        except Exception as e:
            logger.error(f"Error verifying model integrity: {e}")
            return False

    def load_model(self, model_name: str) -> tuple[AutoModelForSeq2SeqGeneration, AutoTokenizer]:
        """
        Load model and tokenizer from local storage
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model_path = self.get_model_path(model_name)
        if not model_path or not model_path.exists():
            raise ValueError(f"Model {model_name} not available locally")

        try:
            model = AutoModelForSeq2SeqGeneration.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            return model, tokenizer
        
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

    def cleanup_old_models(self, keep_days: int = 30):
        """Clean up old unused models"""
        current_time = datetime.utcnow()
        models_to_remove = []
        
        for model_name, info in self.models_info.items():
            last_verified = datetime.fromisoformat(info['last_verified'])
            days_since_verified = (current_time - last_verified).days
            
            if days_since_verified > keep_days:
                models_to_remove.append(model_name)
        
        for model_name in models_to_remove:
            self.remove_model(model_name)

    def remove_model(self, model_name: str):
        """Remove a model from local storage"""
        model_path = self.get_model_path(model_name)
        if model_path and model_path.exists():
            try:
                shutil.rmtree(model_path)
                del self.models_info[model_name]
                self._save_models_info()
                logger.info(f"Model {model_name} removed successfully")
            except Exception as e:
                logger.error(f"Error removing model {model_name}: {e}")

    def _calculate_directory_hash(self, directory: Path) -> str:
        """Calculate hash of directory contents"""
        sha256_hash = hashlib.sha256()

        for filepath in sorted(Path(directory).rglob('*')):
            if filepath.is_file():
                with open(filepath, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def _get_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes"""
        return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())

    def get_models_status(self) -> Dict[str, Any]:
        """Get status information about all managed models"""
        return {
            'models': self.models_info,
            'total_size': sum(info['size'] for info in self.models_info.values()),
            'models_dir': str(self.models_dir),
            'cache_dir': str(self.cache_dir)
        }