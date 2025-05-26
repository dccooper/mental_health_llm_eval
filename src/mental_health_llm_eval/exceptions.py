"""
Exception hierarchy for the mental health LLM evaluation framework.
Provides structured error handling and clear error categorization.
"""

from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EvalError(Exception):
    """Base exception for all evaluation errors."""
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)

class ModelError(EvalError):
    """Raised when there are issues with model interactions."""
    pass

class ConfigError(EvalError):
    """Raised when there are configuration issues."""
    pass

class ValidationError(EvalError):
    """Raised when input validation fails."""
    pass

class SafetyError(EvalError):
    """Raised when safety checks fail."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, severity=ErrorSeverity.CRITICAL, context=context)

class RateLimitError(EvalError):
    """Raised when rate limits are exceeded."""
    pass 