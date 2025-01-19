# Path: codeSage.be/src/models/model_catalog.py

from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from pathlib import Path

@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    task: str
    size: str
    description: str
    requirements: Dict[str, str]
    performance_metrics: Dict[str, float]
    is_default: bool = False

class ModelCatalog:
    """Catalog of available models and their configurations"""
    
    DEFAULT_MODELS = {
        "code-analysis": {
            "name": "Salesforce/codet5-base",
            "task": "code-analysis",
            "size": "900MB",
            "description": "Base model for code understanding and analysis",
            "requirements": {
                "min_ram": "4GB",
                "min_vram": "2GB",
                "disk_space": "2GB"
            },
            "performance_metrics": {
                "accuracy": 0.85,
                "latency_ms": 150
            },
            "is_default": True
        },
        "bug-detection": {
            "name": "Salesforce/codet5-large",
            "task": "bug-detection",
            "size": "1.5GB",
            "description": "Specialized model for bug detection and fixes",
            "requirements": {
                "min_ram": "8GB",
                "min_vram": "4GB",
                "disk_space": "3GB"
            },
            "performance_metrics": {
                "accuracy": 0.88,
                "latency_ms": 200
            }
        }
    }

    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self._load_default_models()

    def _load_default_models(self):
        """Load default models into catalog"""
        for task, info in self.DEFAULT_MODELS.items():
            self.models[task] = ModelInfo(**info)

    def get_model_info(self, task: str) -> Optional[ModelInfo]:
        """Get model information for a specific task"""
        return self.models.get(task)

    def get_default_model(self, task: str) -> Optional[ModelInfo]:
        """Get default model for a specific task"""
        for model in self.models.values():
            if model.task == task and model.is_default:
                return model
        return None

    def get_all_models(self) -> List[ModelInfo]:
        """Get all available models"""
        return list(self.models.values())