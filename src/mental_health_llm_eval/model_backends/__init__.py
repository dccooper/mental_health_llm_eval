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

from typing import Protocol, TypeVar, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

T = TypeVar('T', bound='ModelBackend')

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

class ModelBackend(Protocol):
    """Protocol defining the interface for model backends."""
    
    def initialize(self: T, config: Dict[str, Any]) -> None:
        """
        Initialize the model backend with configuration.
        
        Args:
            config: Backend-specific configuration
        """
        ...
    
    def validate_config(self: T, config: Dict[str, Any]) -> None:
        """
        Validate backend configuration.
        
        Args:
            config: Configuration to validate
        """
        ...
    
    def query(
        self: T,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
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
        ...
    
    def get_capabilities(self: T) -> Dict[str, Any]:
        """
        Get backend capabilities and limitations.
        
        Returns:
            Dictionary of capabilities
        """
        ...

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