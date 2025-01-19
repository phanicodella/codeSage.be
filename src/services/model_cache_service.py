# Path: codeSage.be/src/services/model_cache_service.py

import os
import shutil
import logging
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ModelCacheService:
    def __init__(self, cache_dir: str, max_size_mb: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'models': {},
                'last_cleanup': datetime.now().isoformat()
            }

    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get cached model path if exists"""
        model_hash = self._get_model_hash(model_name)
        model_dir = self.cache_dir / model_hash
        
        if model_dir.exists():
            self.metadata['models'][model_hash]['last_accessed'] = datetime.now().isoformat()
            self._save_metadata()
            return model_dir
        return None

    def cache_model(self, model_name: str, model_files: Dict[str, bytes]):
        """Cache model files"""
        model_hash = self._get_model_hash(model_name)
        model_dir = self.cache_dir / model_hash
        
        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model files
        total_size = 0
        for filename, content in model_files.items():
            file_path = model_dir / filename
            with open(file_path, 'wb') as f:
                f.write(content)
            total_size += len(content)

        # Update metadata  
        self.metadata['models'][model_hash] = {
            'name': model_name,
            'size': total_size,
            'cached_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat()
        }
        self._save_metadata()

        # Run cleanup if needed
        self._cleanup_if_needed()

    def _cleanup_if_needed(self):
        """Clean up old cache entries if size limit exceeded"""
        total_size = sum(m['size'] for m in self.metadata['models'].values())
        
        if total_size > (self.max_size_mb * 1024 * 1024):
            # Sort models by last accessed time
            sorted_models = sorted(
                self.metadata['models'].items(),
                key=lambda x: datetime.fromisoformat(x[1]['last_accessed'])
            )
            
            # Remove oldest models until under limit
            while total_size > (self.max_size_mb * 1024 * 1024):
                if not sorted_models:
                    break
                    
                model_hash, info = sorted_models.pop(0)
                model_dir = self.cache_dir / model_hash
                
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                    total_size -= info['size']
                    del self.metadata['models'][model_hash]

            self._save_metadata()

    def _get_model_hash(self, model_name: str) -> str:
        """Generate deterministic hash for model name"""
        return hashlib.sha256(model_name.encode()).hexdigest()[:12]

    def clear_cache(self):
        """Clear entire cache"""
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True)
        self.metadata = {
            'models': {},
            'last_cleanup': datetime.now().isoformat()
        }
        self._save_metadata()