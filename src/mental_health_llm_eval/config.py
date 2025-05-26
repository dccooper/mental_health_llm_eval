#!/usr/bin/env python3

# config.py — Centralized config for model backend

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from .exceptions import ConfigError

@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for model backends."""
    backend_type: str
    api_key: Optional[str]
    model_name: str
    max_tokens: int
    temperature: float
    organization_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Create configuration from environment variables."""
        load_dotenv()

        backend = os.getenv("MODEL_BACKEND", "openai").lower()
        api_key = os.getenv(f"{backend.upper()}_API_KEY")

        if not api_key:
            raise ConfigError(f"Missing API key for {backend} backend")

        return cls(
            backend_type=backend,
            api_key=api_key,
            model_name=os.getenv("MODEL_NAME", "gpt-4"),
            max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            organization_id=os.getenv(f"{backend.upper()}_ORG_ID")
        )

    def validate(self) -> None:
        """Validate configuration values."""
        if not self.api_key:
            raise ConfigError("API key cannot be empty")
        
        if self.max_tokens < 1:
            raise ConfigError("max_tokens must be positive")
        
        if not 0 <= self.temperature <= 2:
            raise ConfigError("temperature must be between 0 and 2")

@dataclass(frozen=True)
class EvalConfig:
    """Configuration for the evaluation framework."""
    model: ModelConfig
    log_dir: str = "logs"
    batch_size: int = 10
    max_retries: int = 3
    timeout: float = 30.0

    def validate(self) -> None:
        """Validate evaluation configuration."""
        self.model.validate()
        
        if not self.log_dir:
            raise ConfigError("log_dir cannot be empty")
        
        if self.batch_size < 1:
            raise ConfigError("batch_size must be positive")
        
        if self.max_retries < 0:
            raise ConfigError("max_retries cannot be negative")
        
        if self.timeout <= 0:
            raise ConfigError("timeout must be positive")
