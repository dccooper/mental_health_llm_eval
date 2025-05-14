"""
Mental Health LLM Evaluator
=========================

A framework for evaluating LLM responses in mental health contexts.
"""

from importlib.metadata import version

__version__ = version("mental_health_llm_eval")

from .evaluator import Evaluator, EvaluationResult
from .model_backends import ModelBackend, ModelBackendType, ModelResponse
from .validation import ValidationLevel
from .rate_limiter import RateLimiter, LimitType
from .logging_handler import LogManager, ErrorSeverity

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "ModelBackend",
    "ModelBackendType",
    "ModelResponse",
    "ValidationLevel",
    "RateLimiter",
    "LimitType",
    "LogManager",
    "ErrorSeverity",
] 