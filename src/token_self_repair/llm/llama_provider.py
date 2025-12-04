"""
Llama model provider with logit extraction capabilities.

This module provides a clean interface for loading and using Llama models
with LogTokU uncertainty estimation. Refactored from test_llama_simple.py.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LLMClient, TokenLogit


class LlamaProvider(LLMClient):
    """
    Llama model provider with logit extraction.
    
    Supports:
    - Llama-2 family (7B, 13B, 70B)
    - Llama-3 family (8B, 70B)
    - 4-bit quantization for memory efficiency
    - CPU and GPU inference
    
    Example:
        >>> provider = LlamaProvider("meta-llama/Llama-2-7b-chat-hf")
        >>> tokens, logits = provider.generate_with_logits("What is 2+2?")
        >>> print(tokens.shape, logits.shape)
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        use_quantization: bool = False,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize Llama provider.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device to load model on (auto-detected if None)
            use_quantization: Use 4-bit quantization (requires GPU)
            torch_dtype: Torch dtype for model (auto if None)
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_quantization = use_quantization
        
        # Determine dtype
        if torch_dtype is None:
            if self.device.type == 'cuda':
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype
        
        # Load model and tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer."""
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=token
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    
    def _load_model(self) -> AutoModelForCausalLM:
        """Load and configure model."""
        if self.use_quantization and self.device.type == 'cuda':
            return self._load_quantized_model()
        else:
            return self._load_standard_model()
    
    def _load_quantized_model(self) -> AutoModelForCausalLM:
        """Load model with 4-bit quantization."""
        from transformers import BitsAndBytesConfig
        from huggingface_hub import HfFolder
        
        token = HfFolder.get_token()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
        
        model.eval()
        return model
    
    def _load_standard_model(self) -> AutoModelForCausalLM:
        """Load model without quantization."""
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        
        if self.device.type == 'cuda':
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                token=token
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                token=token
            )
            model = model.to(self.device)
        
        model.eval()
        return model
    
    def _format_prompt(self, prompt: str) -> str:
        """
        Format prompt for Llama models.
        
        Handles different Llama formats:
        - Llama-2: Uses [INST] tags
        - Llama-3: Uses chat template
        - TinyLlama-Chat: Uses chat template if available
        - Other: Returns prompt as-is
        """
        model_name_lower = self.model.config.name_or_path.lower()
        
        if 'llama-2' in model_name_lower or 'llama2' in model_name_lower:
            # Llama-2 format
            return f"<s>[INST] {prompt} [/INST]"
        
        elif 'llama-3' in model_name_lower or 'llama3' in model_name_lower:
            # Llama-3 chat template
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        elif 'tinylama' in model_name_lower and 'chat' in model_name_lower:
            # TinyLlama-Chat format - try chat template first
            try:
                messages = [
                    {"role": "user", "content": prompt}
                ]
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            except (KeyError, AttributeError):
                # Fallback to simple format
                return f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        else:
            # Generic format - try chat template if available
            try:
                if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                    messages = [{"role": "user", "content": prompt}]
                    return self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
            except (KeyError, AttributeError, TypeError):
                pass
            return prompt
    
    def generate_with_logits(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate text and return both tokens and logits.
        
        This is the core method for uncertainty estimation, providing
        the raw logits needed for LogTokU calculation.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (if do_sample=True)
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
        
        Returns:
            Tuple of (generated_tokens, logits_tensor)
            - generated_tokens: Shape (seq_len,)
            - logits_tensor: Shape (num_new_tokens, vocab_size)
        
        Example:
            >>> tokens, logits = provider.generate_with_logits("Hello")
            >>> print(f"Generated {len(tokens)} tokens")
            >>> print(f"Logits shape: {logits.shape}")
        """
        # Format prompt
        formatted_prompt = self._format_prompt(prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Get pad token ID
        pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.unk_token_id
        
        # Generate with logits
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=pad_token_id,
                **kwargs
            )
        
        # Extract generated tokens (excluding input)
        generated_tokens = outputs.sequences[0][input_ids.shape[1]:]
        
        # Stack logits: tuple of (vocab_size,) -> (num_tokens, vocab_size)
        logits_list = [score[0].cpu() for score in outputs.scores]
        logits_tensor = torch.stack(logits_list, dim=0)
        
        return generated_tokens, logits_tensor
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        """
        Generate text without logits (standard generation).
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text string
        """
        tokens, _ = self.generate_with_logits(prompt, max_new_tokens, **kwargs)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs
    ):
        """Yield TokenLogit objects for streaming pipelines."""

        token_ids, logits_tensor = self.generate_with_logits(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
            **kwargs,
        )

        for token_id, logit_row in zip(token_ids, logits_tensor):
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            if token_text == "":
                # Fallback to cleaned decode if tokenizer produced empty string
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            yield TokenLogit(token=token_text, logits=logit_row.tolist())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'quantized': self.use_quantization,
            'dtype': str(self.torch_dtype),
            'vocab_size': self.tokenizer.vocab_size,
            'model_size': sum(p.numel() for p in self.model.parameters())
        }
    
    def __repr__(self) -> str:
        info = self.get_model_info()
        return (
            f"LlamaProvider(\n"
            f"  model={info['model_name']},\n"
            f"  device={info['device']},\n"
            f"  quantized={info['quantized']},\n"
            f"  vocab_size={info['vocab_size']:,}\n"
            f")"
        )


# Convenience function for quick testing
def load_llama(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantize: bool = False
) -> LlamaProvider:
    """
    Quick loader for Llama models.
    
    Args:
        model_name: Model to load (defaults to TinyLlama for testing)
        quantize: Use 4-bit quantization
    
    Returns:
        Configured LlamaProvider
    
    Example:
        >>> provider = load_llama()
        >>> text = provider.generate("What is AI?")
    """
    return LlamaProvider(model_name, use_quantization=quantize)

