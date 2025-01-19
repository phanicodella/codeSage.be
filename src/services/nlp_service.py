# Path: codeSage.be/src/services/nlp_service.py

import torch
from transformers import AutoModelForSeq2SeqGeneration, pipeline
from typing import List, Dict, Union, Optional
import logging
from pathlib import Path
import os
import json
from concurrent.futures import ThreadPoolExecutor
from ..models.tokenizer import CodeTokenizer

logger = logging.getLogger(__name__)

class NLPService:
    """
    Service for handling natural language processing tasks related to code analysis,
    including code understanding, bug detection, and code generation.
    """

    def __init__(self, 
                 model_name: str = "Salesforce/codet5-base",
                 device: str = None,
                 max_workers: int = 2):
        """
        Initialize the NLP service with specified model and configurations.

        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cuda' or 'cpu')
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.model_name = model_name
        self.max_workers = max_workers
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self._initialize_model()
            self.tokenizer = CodeTokenizer(model_name)
            logger.info(f"Successfully initialized NLP service with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize NLP service: {str(e)}")
            raise

    def _initialize_model(self):
        """Initialize the model and move it to the appropriate device"""
        try:
            # Try loading from local cache first
            cache_dir = Path(os.getenv('MODEL_CACHE_DIR', './models'))
            self.model = AutoModelForSeq2SeqGeneration.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.warning(f"Could not load model from cache: {str(e)}")
            # Download and save to cache
            self.model = AutoModelForSeq2SeqGeneration.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
        
        self.model.to(self.device)
        self.model.eval()

    def analyze_code_block(self, 
                          code: str, 
                          task: str = "understand",
                          max_length: int = 512) -> Dict[str, Union[str, float]]:
        """
        Analyze a block of code for understanding, bugs, or improvements.

        Args:
            code: The code block to analyze
            task: Type of analysis ("understand", "detect_bugs", "improve")
            max_length: Maximum length of generated response

        Returns:
            Dictionary containing analysis results
        """
        task_prefixes = {
            "understand": "Explain the following code:\n",
            "detect_bugs": "Identify potential bugs in:\n",
            "improve": "Suggest improvements for:\n"
        }
        
        prefix = task_prefixes.get(task, task_prefixes["understand"])
        input_text = f"{prefix}{code}"
        
        try:
            inputs = self.tokenizer.encode(input_text, max_length=max_length)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'].to(self.device),
                    attention_mask=inputs['attention_mask'].to(self.device),
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            decoded_output = self.tokenizer.decode(outputs[0])
            
            return {
                "task": task,
                "analysis": decoded_output,
                "confidence": self._calculate_confidence(outputs)
            }
        except Exception as e:
            logger.error(f"Error analyzing code block: {str(e)}")
            return {
                "task": task,
                "analysis": "Error analyzing code",
                "error": str(e),
                "confidence": 0.0
            }

    def batch_analyze_code(self, 
                          code_blocks: List[str],
                          task: str = "understand") -> List[Dict[str, Union[str, float]]]:
        """
        Analyze multiple code blocks in parallel.

        Args:
            code_blocks: List of code blocks to analyze
            task: Type of analysis to perform

        Returns:
            List of analysis results for each code block
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.analyze_code_block, code, task)
                for code in code_blocks
            ]
            return [future.result() for future in futures]

    def _calculate_confidence(self, outputs: torch.Tensor) -> float:
        """
        Calculate confidence score for model outputs.

        Args:
            outputs: Model output tensor

        Returns:
            Confidence score between 0 and 1
        """
        try:
            logits = outputs.scores[0] if hasattr(outputs, 'scores') else None
            if logits is not None:
                probs = torch.softmax(logits, dim=-1)
                return float(torch.max(probs).mean())
            return 0.8  # Default confidence when scores aren't available
        except Exception:
            return 0.5

    def save_analysis_cache(self, 
                          cache_file: str,
                          analysis_results: Dict[str, Dict]):
        """
        Save analysis results to cache file.

        Args:
            cache_file: Path to cache file
            analysis_results: Dictionary of analysis results to cache
        """
        try:
            with open(cache_file, 'w') as f:
                json.dump(analysis_results, f)
        except Exception as e:
            logger.error(f"Error saving analysis cache: {str(e)}")

    def load_analysis_cache(self, cache_file: str) -> Optional[Dict]:
        """
        Load cached analysis results.

        Args:
            cache_file: Path to cache file

        Returns:
            Dictionary of cached analysis results or None if cache doesn't exist
        """
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading analysis cache: {str(e)}")
        return None

    def clear_gpu_memory(self):
        """Clear GPU memory if using CUDA device"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup when the service is destroyed"""
        self.clear_gpu_memory()