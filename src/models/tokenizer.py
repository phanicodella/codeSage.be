# Path: codeSage.be/src/models/tokenizer.py

from transformers import AutoTokenizer, PreTrainedTokenizerFast
import torch
from typing import List, Union, Dict
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class CodeTokenizer:
    """
    Handles tokenization of code for various programming languages using CodeT5 tokenizer.
    Provides methods for both encoding and decoding text, with special handling for
    code-specific tokens and multiple programming languages.
    """

    def __init__(self, model_name: str = "Salesforce/codet5-base"):
        """
        Initialize the tokenizer with a specific model.
        
        Args:
            model_name (str): Name of the pre-trained model to use for tokenization
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True
            )
            logger.info(f"Successfully loaded tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            # Fallback to local cached version if available
            cache_dir = Path(os.getenv('MODEL_CACHE_DIR', './models'))
            if (cache_dir / model_name).exists():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    cache_dir / model_name,
                    trust_remote_code=True,
                    local_files_only=True
                )
            else:
                raise RuntimeError(f"Could not load tokenizer and no local cache found: {str(e)}")

    def encode(self, text: Union[str, List[str]], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Encode text into tokens.

        Args:
            text: Single string or list of strings to tokenize
            max_length: Maximum length of the tokenized sequence

        Returns:
            Dictionary containing input_ids, attention_mask, and other relevant tensors
        """
        try:
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            return encoded
        except Exception as e:
            logger.error(f"Encoding failed: {str(e)}")
            raise

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token IDs back into text.

        Args:
            token_ids: Tensor of token IDs to decode
            skip_special_tokens: Whether to remove special tokens in the decoding

        Returns:
            Decoded text as string or list of strings
        """
        try:
            if token_ids.dim() == 1:
                return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
        except Exception as e:
            logger.error(f"Decoding failed: {str(e)}")
            raise

    def get_special_tokens(self) -> Dict[str, str]:
        """
        Get the special tokens used by the tokenizer.

        Returns:
            Dictionary mapping special token names to their string values
        """
        return {
            'pad_token': self.tokenizer.pad_token,
            'eos_token': self.tokenizer.eos_token,
            'bos_token': self.tokenizer.bos_token,
            'unk_token': self.tokenizer.unk_token,
            'mask_token': self.tokenizer.mask_token
        }

    def get_vocab_size(self) -> int:
        """
        Get the size of the tokenizer's vocabulary.

        Returns:
            Integer representing the vocabulary size
        """
        return self.tokenizer.vocab_size

    def tokenize_code_snippet(self, code: str, language: str = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize a code snippet with optional language-specific handling.

        Args:
            code: The code snippet to tokenize
            language: Programming language of the code (optional)

        Returns:
            Dictionary containing tokenized representation of the code
        """
        if language:
            # Add language-specific token if available
            code = f"<{language}> {code}"
        
        return self.encode(code)

    @property
    def max_len(self) -> int:
        """
        Get the maximum sequence length supported by the tokenizer.

        Returns:
            Integer representing maximum sequence length
        """
        return self.tokenizer.model_max_length