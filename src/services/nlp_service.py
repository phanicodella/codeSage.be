# Path: codeSage.be/src/services/nlp_service.py

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, PreTrainedModel
from typing import List, Dict, Union, Optional, Tuple
import logging
from pathlib import Path
import os
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading
import queue
from ..models.tokenizer import CodeTokenizer
import gc

logger = logging.getLogger(__name__)

class NLPService:
    """
    Production-ready service for handling natural language processing tasks related to code analysis,
    including code understanding, bug detection, and code generation.
    """
    
    # Class-level lock for thread-safe model operations
    _model_lock = threading.Lock()
    
    # Queue for batch processing
    _request_queue = queue.Queue()
    
    def __init__(self, 
                 model_name: str = "Salesforce/codet5-base",
                 device: str = None,
                 max_workers: int = 2,
                 batch_size: int = 8,
                 max_sequence_length: int = 512,
                 cache_dir: Optional[str] = None,
                 model_revision: str = "main"):
        """
        Initialize the NLP service with specified model and configurations.

        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cuda' or 'cpu')
            max_workers: Maximum number of worker threads for parallel processing
            batch_size: Size of batches for processing multiple requests
            max_sequence_length: Maximum sequence length for tokenization
            cache_dir: Directory to cache models
            model_revision: Model revision/version to use
        """
        self.model_name = model_name
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.model_revision = model_revision
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir or os.getenv('MODEL_CACHE_DIR', './models'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and tokenizer
        self._initialize_services()
        
        # Start batch processing worker
        self._start_batch_worker()
        
        # Track model usage metrics
        self.metrics = {
            'requests_processed': 0,
            'last_batch_time': None,
            'average_processing_time': 0,
            'errors': 0
        }
        
        logger.info(f"NLP Service initialized with model: {model_name} on device: {self.device}")

    def _initialize_services(self):
        """Initialize model and tokenizer with error handling and retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._initialize_model()
                self.tokenizer = CodeTokenizer(self.model_name)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to initialize NLP services after {max_retries} attempts")
                    raise
                logger.warning(f"Initialization attempt {attempt + 1} failed: {str(e)}. Retrying...")
                self._cleanup()

    def _initialize_model(self) -> None:
        """Initialize the model with proper error handling and logging"""
        try:
            with self._model_lock:
                model_path = self.cache_dir / self.model_name.replace('/', '_')
                
                if model_path.exists():
                    logger.info("Loading model from local cache")
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                else:
                    logger.info("Downloading model from Hugging Face")
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        cache_dir=self.cache_dir,
                        revision=self.model_revision
                    )
                    # Save model locally
                    self.model.save_pretrained(model_path)
                
                self.model.to(self.device)
                self.model.eval()
                
                # Enable gradient checkpointing for memory efficiency
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()
                
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    async def analyze_code_block(self, 
                               code: str, 
                               task: str = "understand",
                               max_length: Optional[int] = None) -> Dict[str, Union[str, float]]:
        """
        Analyze a block of code with proper error handling and resource management.

        Args:
            code: The code block to analyze
            task: Type of analysis ("understand", "detect_bugs", "improve")
            max_length: Maximum length of generated response

        Returns:
            Dictionary containing analysis results and confidence scores
        """
        if not code.strip():
            raise ValueError("Empty code block provided")
            
        max_length = max_length or self.max_sequence_length
        task_prefixes = {
            "understand": "Explain the following code:\n",
            "detect_bugs": "Identify potential bugs in:\n",
            "improve": "Suggest improvements for:\n"
        }
        
        prefix = task_prefixes.get(task, task_prefixes["understand"])
        input_text = f"{prefix}{code}"
        
        try:
            with self._model_lock:
                with torch.no_grad():
                    inputs = self.tokenizer.encode(
                        input_text, 
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=max_length,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.95,
                        repetition_penalty=1.2,
                        early_stopping=True
                    )
                    
                    decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Calculate confidence scores
                    confidence = self._calculate_confidence(outputs)
                    
                    self.metrics['requests_processed'] += 1
                    
                    return {
                        "task": task,
                        "analysis": decoded_output,
                        "confidence": confidence,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error analyzing code block: {str(e)}")
            raise RuntimeError(f"Analysis failed: {str(e)}")
        finally:
            # Clean up GPU memory if needed
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def batch_analyze_code(self, 
                          code_blocks: List[str],
                          task: str = "understand") -> List[Dict[str, Union[str, float]]]:
        """Process multiple code blocks in parallel with batching"""
        results = []
        for i in range(0, len(code_blocks), self.batch_size):
            batch = code_blocks[i:i + self.batch_size]
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                batch_results = list(executor.map(
                    lambda code: self.analyze_code_block(code, task),
                    batch
                ))
                results.extend(batch_results)
        return results

    def _calculate_confidence(self, outputs: torch.Tensor) -> float:
        """Calculate confidence score for model outputs"""
        try:
            if hasattr(outputs, 'scores'):
                logits = outputs.scores[0]
                probs = torch.softmax(logits, dim=-1)
                return float(torch.max(probs).mean())
            return self._calculate_alternative_confidence(outputs)
        except Exception:
            return 0.5

    def _calculate_alternative_confidence(self, outputs: torch.Tensor) -> float:
        """Alternative confidence calculation when scores aren't available"""
        try:
            # Use output probabilities distribution as a proxy for confidence
            last_hidden = outputs.last_hidden_state
            variance = torch.var(last_hidden, dim=-1).mean()
            return float(1.0 / (1.0 + variance))
        except Exception:
            return 0.5

    def _start_batch_worker(self) -> None:
        """Start background worker for processing batched requests"""
        def worker():
            while True:
                try:
                    batch = []
                    try:
                        while len(batch) < self.batch_size:
                            item = self._request_queue.get_nowait()
                            batch.append(item)
                    except queue.Empty:
                        pass
                    
                    if batch:
                        start_time = datetime.now()
                        self._process_batch(batch)
                        self.metrics['last_batch_time'] = (datetime.now() - start_time).total_seconds()
                except Exception as e:
                    logger.error(f"Batch processing error: {str(e)}")
                    
        threading.Thread(target=worker, daemon=True).start()

    def _process_batch(self, batch: List[Dict]) -> None:
        """Process a batch of requests"""
        try:
            with self._model_lock:
                for item in batch:
                    code = item['code']
                    task = item.get('task', 'understand')
                    callback = item.get('callback')
                    
                    try:
                        result = self.analyze_code_block(code, task)
                        if callback:
                            callback(result)
                    except Exception as e:
                        if callback:
                            callback({'error': str(e)})
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")

    def get_metrics(self) -> Dict[str, Union[int, float, str]]:
        """Get service metrics"""
        return {
            **self.metrics,
            'device': self.device,
            'model_name': self.model_name,
            'queue_size': self._request_queue.qsize()
        }

    def _cleanup(self) -> None:
        """Clean up resources"""
        try:
            with self._model_lock:
                del self.model
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    def __del__(self):
        """Cleanup on destruction"""
        self._cleanup()