"""
Custom Model Backend Template
=========================

Template for implementing custom model backends.
Users can extend this class to integrate their own models.
"""

from typing import Dict, Any, List, Optional
from . import ModelBackend, ModelResponse

class CustomBackend(ModelBackend):
    """Template for custom model backend implementation."""
    
    def __init__(self):
        """Initialize custom backend."""
        self.model = None
        self.initialized = False
        # Add any custom initialization here
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the custom backend.
        
        Args:
            config: Configuration dictionary with custom parameters
        """
        errors = self.validate_config(config)
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
        
        # Example: Load your custom model
        # self.model = YourCustomModel(**config)
        
        self.initialized = True
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate custom configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Add your custom config validation here
        # Example:
        # if "model_path" not in config:
        #     errors.append("Missing model path")
        
        return errors
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Query custom model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional custom parameters
                
        Returns:
            Structured model response
        """
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
        
        try:
            # Example implementation:
            # 1. Prepare input
            # full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            # 2. Run inference
            # result = self.model.generate(full_prompt, **kwargs)
            
            # 3. Process output
            # response_text = post_process(result)
            
            # 4. Collect metadata
            # metadata = {
            #     "model": "custom_model",
            #     "custom_field": "custom_value"
            # }
            
            # For template, return empty response
            return ModelResponse(
                text="Custom backend not implemented",
                metadata={},
                error="This is a template. Implement your custom logic."
            )
            
        except Exception as e:
            return ModelResponse(
                text="",
                metadata={},
                error=str(e)
            )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get custom backend capabilities.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "streaming": False,  # Update based on your implementation
            "system_prompts": True,  # Update based on your implementation
            "function_calling": False,  # Update based on your implementation
            "custom_capability": "custom_value"  # Add your custom capabilities
        } 