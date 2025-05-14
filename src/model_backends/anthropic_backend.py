"""
Anthropic Model Backend
====================

Implementation of the Anthropic API backend for model queries.
Supports Claude and other Anthropic models.
"""

import os
from typing import Dict, Any, List, Optional
import anthropic
from . import ModelBackend, ModelResponse

class AnthropicBackend(ModelBackend):
    """Anthropic API backend implementation."""
    
    def __init__(self):
        """Initialize Anthropic backend."""
        self.client = None
        self.default_model = None
        self.initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Anthropic backend.
        
        Args:
            config: Configuration dictionary with:
                - api_key: Anthropic API key
                - model: Default model to use (e.g., "claude-2")
        """
        errors = self.validate_config(config)
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
        
        self.client = anthropic.Client(api_key=config["api_key"])
        self.default_model = config.get("model", "claude-2")
        self.initialized = True
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate Anthropic configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if "api_key" not in config:
            errors.append("Missing Anthropic API key")
        elif not isinstance(config["api_key"], str):
            errors.append("API key must be a string")
        
        if "model" in config:
            if not isinstance(config["model"], str):
                errors.append("Model must be a string")
            elif not config["model"].startswith("claude"):
                errors.append("Model must be a Claude model")
        
        return errors
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Query Anthropic API.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters:
                - model: Override default model
                - max_tokens_to_sample: Maximum tokens in response
                - temperature: Sampling temperature
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                
        Returns:
            Structured model response
        """
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        try:
            # Prepare prompt
            if system_prompt:
                full_prompt = f"{anthropic.HUMAN_PROMPT} {system_prompt}\n\n{prompt}{anthropic.AI_PROMPT}"
            else:
                full_prompt = f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}"
            
            # Extract Anthropic-specific parameters
            model = kwargs.pop("model", self.default_model)
            
            # Call API
            response = self.client.completion(
                prompt=full_prompt,
                model=model,
                **kwargs
            )
            
            # Extract response
            result = response.completion.strip()
            
            # Collect metadata
            metadata = {
                "model": model,
                "stop_reason": response.stop_reason,
                "usage": {
                    "prompt_tokens": len(full_prompt.split()),  # Approximate
                    "completion_tokens": len(result.split())    # Approximate
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
        Get Anthropic backend capabilities.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "streaming": True,
            "system_prompts": True,
            "function_calling": False,  # Not supported yet
            "max_tokens": {
                "claude-2": 100000,
                "claude-instant-1": 100000
            },
            "supports_models": [
                "claude-2",
                "claude-instant-1",
                "claude-1",
                "claude-1-100k"
            ]
        } 