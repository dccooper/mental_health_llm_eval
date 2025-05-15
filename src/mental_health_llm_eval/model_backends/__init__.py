"""
Model Backends Package
===================

This package provides interfaces and implementations for different LLM backends.
It supports:
1. Public API providers (OpenAI, Anthropic, etc.)
2. Local models (via HuggingFace)
3. Custom in-house models
4. Mock/stub models for testing
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class ModelBackendType(Enum):
    """Types of supported model backends."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"
    MOCK = "mock"

@dataclass
class ModelResponse:
    """Structured response from a model."""
    text: str
    metadata: Dict[str, Any]
    error: Optional[str] = None

class ModelBackend(ABC):
    """Abstract base class for model backends."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the model backend with configuration.
        
        Args:
            config: Backend-specific configuration
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate backend configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        pass
    
    @abstractmethod
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Query the model with a prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional backend-specific parameters
            
        Returns:
            Structured model response
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get backend capabilities and limitations.
        
        Returns:
            Dictionary of capabilities
        """
        pass

def create_backend(
    backend_type: ModelBackendType,
    config: Dict[str, Any]
) -> ModelBackend:
    """
    Factory function to create model backends.
    
    Args:
        backend_type: Type of backend to create
        config: Backend configuration
        
    Returns:
        Configured model backend instance
    """
    from .openai_backend import OpenAIBackend
    from .anthropic_backend import AnthropicBackend
    from .huggingface_backend import HuggingFaceBackend
    from .custom_backend import CustomBackend
    from .mock_backend import MockBackend
    
    backends = {
        ModelBackendType.OPENAI: OpenAIBackend,
        ModelBackendType.ANTHROPIC: AnthropicBackend,
        ModelBackendType.HUGGINGFACE: HuggingFaceBackend,
        ModelBackendType.CUSTOM: CustomBackend,
        ModelBackendType.MOCK: MockBackend
    }
    
    backend_class = backends.get(backend_type)
    if not backend_class:
        raise ValueError(f"Unsupported backend type: {backend_type}")
    
    backend = backend_class()
    backend.initialize(config)
    return backend 