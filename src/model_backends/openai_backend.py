"""
OpenAI Model Backend
==================

Implementation of the OpenAI API backend for model queries.
Supports GPT-3.5, GPT-4, and other OpenAI models.
"""

import os
from typing import Dict, Any, List, Optional
import openai
from . import ModelBackend, ModelResponse

class OpenAIBackend(ModelBackend):
    """OpenAI API backend implementation."""
    
    def __init__(self):
        """Initialize OpenAI backend."""
        self.api_key = None
        self.default_model = None
        self.initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the OpenAI backend.
        
        Args:
            config: Configuration dictionary with:
                - api_key: OpenAI API key
                - model: Default model to use
                - organization: (optional) OpenAI organization ID
        """
        errors = self.validate_config(config)
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
        
        self.api_key = config["api_key"]
        self.default_model = config.get("model", "gpt-4")
        
        # Configure OpenAI client
        openai.api_key = self.api_key
        if "organization" in config:
            openai.organization = config["organization"]
        
        self.initialized = True
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate OpenAI configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if "api_key" not in config:
            errors.append("Missing OpenAI API key")
        elif not isinstance(config["api_key"], str):
            errors.append("API key must be a string")
        
        if "model" in config and not isinstance(config["model"], str):
            errors.append("Model must be a string")
        
        if "organization" in config and not isinstance(config["organization"], str):
            errors.append("Organization must be a string")
        
        return errors
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Query OpenAI API.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters:
                - model: Override default model
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens in response
                - top_p: Nucleus sampling parameter
                - frequency_penalty: Frequency penalty
                - presence_penalty: Presence penalty
                
        Returns:
            Structured model response
        """
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Extract OpenAI-specific parameters
            model = kwargs.pop("model", self.default_model)
            
            # Call API
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            # Extract response
            result = response['choices'][0]['message']['content'].strip()
            
            # Collect metadata
            metadata = {
                "model": model,
                "finish_reason": response['choices'][0]['finish_reason'],
                "usage": response.get("usage", {}),
                "created": response.get("created")
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
        Get OpenAI backend capabilities.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "streaming": True,
            "system_prompts": True,
            "function_calling": True,
            "max_tokens": {
                "gpt-4": 8192,
                "gpt-3.5-turbo": 4096
            },
            "supports_models": [
                "gpt-4",
                "gpt-4-32k",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k"
            ]
        } 