"""
Mock Model Backend
===============

Mock backend implementation for testing purposes.
Provides deterministic responses based on input prompts.
"""

from typing import Dict, Any, List, Optional
import hashlib
import time
from . import ModelBackend, ModelResponse

class MockBackend(ModelBackend):
    """Mock backend for testing."""
    
    def __init__(self):
        """Initialize mock backend."""
        self.delay = 0
        self.error_rate = 0
        self.initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the mock backend.
        
        Args:
            config: Configuration dictionary with:
                - delay: Artificial delay in seconds
                - error_rate: Probability of error (0-1)
        """
        errors = self.validate_config(config)
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
        
        self.delay = config.get("delay", 0)
        self.error_rate = config.get("error_rate", 0)
        self.initialized = True
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate mock configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if "delay" in config:
            if not isinstance(config["delay"], (int, float)):
                errors.append("Delay must be a number")
            elif config["delay"] < 0:
                errors.append("Delay must be non-negative")
        
        if "error_rate" in config:
            if not isinstance(config["error_rate"], (int, float)):
                errors.append("Error rate must be a number")
            elif not 0 <= config["error_rate"] <= 1:
                errors.append("Error rate must be between 0 and 1")
        
        return errors
    
    def _generate_deterministic_response(self, prompt: str) -> str:
        """
        Generate a deterministic response based on input prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Deterministic response
        """
        # Use hash of prompt to generate deterministic response
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        # Generate mock response based on hash
        responses = [
            "This is a mock response for testing purposes.",
            "The mock backend is working correctly.",
            "Your prompt has been processed by the mock backend.",
            "This response is deterministically generated.",
            "Mock backend simulation complete."
        ]
        
        # Use first 8 chars of hash as integer for indexing
        index = int(prompt_hash[:8], 16) % len(responses)
        return responses[index]
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Query mock model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters (ignored)
                
        Returns:
            Structured model response
        """
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        try:
            # Simulate processing delay
            if self.delay > 0:
                time.sleep(self.delay)
            
            # Simulate random errors
            if self.error_rate > 0:
                import random
                if random.random() < self.error_rate:
                    raise RuntimeError("Simulated random error")
            
            # Generate response
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            response = self._generate_deterministic_response(full_prompt)
            
            # Simulate token counts
            input_tokens = len(full_prompt.split())
            output_tokens = len(response.split())
            
            metadata = {
                "model": "mock_model",
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                "mock_info": {
                    "delay": self.delay,
                    "error_rate": self.error_rate
                }
            }
            
            return ModelResponse(
                text=response,
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
        Get mock backend capabilities.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "streaming": False,
            "system_prompts": True,
            "function_calling": False,
            "is_mock": True,
            "deterministic": True,
            "configurable_delay": True,
            "configurable_error_rate": True
        } 