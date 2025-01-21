from typing import Optional, Dict, Any
import torch
from torch.quantization import quantize_dynamic
from transformers import AutoTokenizer, T5ForConditionalGeneration
from huggingface_hub import snapshot_download
import logging
from pathlib import Path
import tempfile
import shutil
import os
import threading
import queue
import gc

logger = logging.getLogger(__name__)

class NLPService:
    """Service for handling NLP tasks with proper model caching and initialization."""
    
    _model_lock = threading.Lock()
    _request_queue = queue.Queue()

    def __init__(self, 
                 cache_dir: str,
                 model_name: str = "Salesforce/codet5-base",
                 device: Optional[str] = None,
                 enable_quantization: bool = True):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_quantization = enable_quantization
        
        # Create cache directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self._initialize_services()
        
        # Usage metrics
        self.metrics = {
            'requests_processed': 0,
            'last_batch_time': None,
            'average_processing_time': 0,
            'errors': 0
        }

    def _initialize_services(self):
        """Initialize model and tokenizer with error handling and retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._initialize_model()
                self._initialize_tokenizer()
                
                # Apply quantization if enabled
                if self.enable_quantization and self.device == 'cpu':
                    self._quantize_model()
                    
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to initialize NLP services after {max_retries} attempts")
                    raise
                logger.warning(f"Initialization attempt {attempt + 1} failed: {str(e)}. Retrying...")
                self._cleanup()

    def _initialize_model(self):
        """Initialize the model with proper error handling"""
        try:
            with self._model_lock:
                # Ensure model is downloaded and cached
                model_path = self._ensure_model_cached()
                
                # Load model from cache
                self.model = T5ForConditionalGeneration.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                # Move model to device
                self.model.to(self.device)
                self.model.eval()
                
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _ensure_model_cached(self) -> Path:
        """Ensure model is downloaded and properly cached"""
        model_path = self.cache_dir / self.model_name.replace('/', '_')
        
        if not self._is_model_cached(model_path):
            logger.info("Downloading model from Hugging Face")
            try:
                # Download to temporary directory first
                with tempfile.TemporaryDirectory() as temp_dir:
                    snapshot_path = snapshot_download(
                        repo_id=self.model_name,
                        cache_dir=temp_dir,
                        local_files_only=False
                    )
                    # Move to final location
                    if model_path.exists():
                        shutil.rmtree(model_path)
                    shutil.copytree(snapshot_path, model_path)
            except Exception as e:
                logger.error(f"Model download failed: {str(e)}")
                raise
                
        return model_path

    def _is_model_cached(self, path: Path) -> bool:
        """Check if model is properly cached"""
        return path.exists() and (path / 'config.json').exists()

    def _initialize_tokenizer(self):
        """Initialize the tokenizer"""
        try:
            model_path = self.cache_dir / self.model_name.replace('/', '_')
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Tokenizer initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}")

    def _quantize_model(self):
        """Quantize model for reduced memory footprint"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                logger.info("Applying dynamic quantization to model")
                self.model = quantize_dynamic(
                    self.model, 
                    {torch.nn.Linear}, 
                    dtype=torch.qint8
                )
        except Exception as e:
            logger.warning(f"Model quantization failed: {str(e)}")

    async def analyze_code_block(self, code: str, task: str = "understand") -> Dict[str, Any]:
        """Analyze a block of code"""
        if not code.strip():
            raise ValueError("Empty code block provided")

        try:
            with self._model_lock:
                with torch.no_grad():
                    # Prepare input
                    prefix = "Explain the following code:\n" if task == "understand" else code
                    inputs = self.tokenizer(
                        prefix + code,
                        max_length=512,
                        truncation=True,
                        padding='max_length',
                        return_tensors="pt"
                    ).to(self.device)

                    # Generate analysis
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=150,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.95
                    )

                    decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Update metrics
                    self.metrics['requests_processed'] += 1
                    
                    return {
                        "task": task,
                        "analysis": decoded_output,
                        "confidence": 0.85  # Placeholder confidence score
                    }

        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error analyzing code block: {str(e)}")
            raise RuntimeError(f"Analysis failed: {str(e)}")
        finally:
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def _cleanup(self):
        """Clean up resources"""
        try:
            with self._model_lock:
                if hasattr(self, 'model'):
                    del self.model
                if hasattr(self, 'tokenizer'):
                    del self.tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    def __del__(self):
        """Cleanup on destruction"""
        self._cleanup()