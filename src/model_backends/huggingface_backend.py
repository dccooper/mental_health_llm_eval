"""
HuggingFace Model Backend
======================

Implementation of the HuggingFace backend for local model inference.
Supports both local models and the HuggingFace API.
"""

import os
from typing import Dict, Any, List, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextGenerationPipeline
)
from . import ModelBackend, ModelResponse

class HuggingFaceBackend(ModelBackend):
    """HuggingFace backend implementation."""
    
    def __init__(self):
        """Initialize HuggingFace backend."""
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = None
        self.device = None
        self.initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the HuggingFace backend.
        
        Args:
            config: Configuration dictionary with:
                - model_name: Model name or path
                - device: Device to run on ("cpu", "cuda", "mps")
                - load_in_8bit: Whether to load in 8-bit mode
                - torch_dtype: Torch dtype for model
                - use_auth_token: HuggingFace auth token for private models
        """
        errors = self.validate_config(config)
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
        
        self.model_name = config["model_name"]
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=config.get("use_auth_token")
        )
        
        # Load model with optimizations
        model_kwargs = {}
        if config.get("load_in_8bit"):
            model_kwargs["load_in_8bit"] = True
        if "torch_dtype" in config:
            model_kwargs["torch_dtype"] = getattr(torch, config["torch_dtype"])
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            use_auth_token=config.get("use_auth_token"),
            **model_kwargs
        )
        
        # Create generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        self.initialized = True
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate HuggingFace configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if "model_name" not in config:
            errors.append("Missing model name")
        elif not isinstance(config["model_name"], str):
            errors.append("Model name must be a string")
        
        if "device" in config:
            if not isinstance(config["device"], str):
                errors.append("Device must be a string")
            elif config["device"] not in ["cpu", "cuda", "mps"]:
                errors.append("Device must be 'cpu', 'cuda', or 'mps'")
        
        if "load_in_8bit" in config and not isinstance(config["load_in_8bit"], bool):
            errors.append("load_in_8bit must be a boolean")
        
        if "torch_dtype" in config and not isinstance(config["torch_dtype"], str):
            errors.append("torch_dtype must be a string")
        
        if "use_auth_token" in config and not isinstance(config["use_auth_token"], str):
            errors.append("use_auth_token must be a string")
        
        return errors
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Query local HuggingFace model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters:
                - max_new_tokens: Maximum new tokens to generate
                - temperature: Sampling temperature
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                - num_return_sequences: Number of sequences to return
                - do_sample: Whether to use sampling
                
        Returns:
            Structured model response
        """
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        try:
            # Prepare prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Get input token count
            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")
            input_token_count = input_ids.shape[1]
            
            # Generate response
            outputs = self.pipeline(
                full_prompt,
                return_full_text=False,  # Only return the generated text
                **kwargs
            )
            
            # Extract response
            result = outputs[0]["generated_text"].strip()
            
            # Get output token count
            output_ids = self.tokenizer.encode(result, return_tensors="pt")
            output_token_count = output_ids.shape[1]
            
            # Collect metadata
            metadata = {
                "model": self.model_name,
                "device": self.device,
                "usage": {
                    "prompt_tokens": input_token_count,
                    "completion_tokens": output_token_count,
                    "total_tokens": input_token_count + output_token_count
                }
            }
            
            return ModelResponse(
                text=result,
                metadata=metadata
            )
            
        except Exception as e:
            return ModelResponse(
                text="",
                metadata={},
                error=str(e)
            )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get HuggingFace backend capabilities.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "streaming": True,
            "system_prompts": True,
            "function_calling": False,
            "supports_8bit": True,
            "supports_4bit": False,  # Not implemented yet
            "supported_devices": [
                "cpu",
                "cuda" if torch.cuda.is_available() else None,
                "mps" if torch.backends.mps.is_available() else None
            ],
            "supported_dtypes": [
                "float32",
                "float16",
                "bfloat16"
            ]
        } 