"""
Logging and Error Handling Module
===============================

This module provides centralized logging and error handling for the mental health LLM evaluation framework.
It includes:
1. Structured logging with different severity levels
2. Custom exception classes for different types of errors
3. Error tracking and aggregation
4. Audit trail functionality
"""

import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import threading
from functools import wraps

# Configure logging format
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"           # Minor issues that don't affect functionality
    MEDIUM = "medium"     # Issues that may affect results but don't break functionality
    HIGH = "high"        # Serious issues that could affect safety or reliability
    CRITICAL = "critical" # Issues that require immediate attention

class EvalError(Exception):
    """Base exception class for evaluation errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        self.message = message
        self.severity = severity
        self.timestamp = datetime.now()
        super().__init__(self.message)

class ValidationError(EvalError):
    """Raised when input validation fails."""
    pass

class ModelError(EvalError):
    """Raised when there are issues with model interactions."""
    pass

class SafetyError(EvalError):
    """Raised when safety checks fail."""
    def __init__(self, message: str):
        super().__init__(message, severity=ErrorSeverity.CRITICAL)

class SystemError(EvalError):
    """Raised for system-level issues."""
    pass

@dataclass
class ErrorContext:
    """Stores context information about an error."""
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime
    stack_trace: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

class LogManager:
    """Manages logging and error tracking for the evaluation framework."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure file handler
        self.log_file = self.log_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        
        # Configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        
        # Configure root logger
        self.logger = logging.getLogger("mental_health_eval")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Error tracking
        self.errors: List[ErrorContext] = []
        self._error_lock = threading.Lock()
        
        # Audit trail
        self.audit_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    def log_error(self, error: Exception, additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error with context.
        
        Args:
            error: The exception that occurred
            additional_info: Additional context about the error
        """
        if isinstance(error, EvalError):
            severity = error.severity
        else:
            severity = ErrorSeverity.MEDIUM
        
        error_context = ErrorContext(
            error_type=error.__class__.__name__,
            message=str(error),
            severity=severity,
            timestamp=datetime.now(),
            stack_trace=logging.traceback.format_exc(),
            additional_info=additional_info
        )
        
        with self._error_lock:
            self.errors.append(error_context)
        
        log_message = f"{error_context.error_type}: {error_context.message}"
        if additional_info:
            log_message += f" | Context: {json.dumps(additional_info)}"
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def log_audit(self, action: str, details: Dict[str, Any]) -> None:
        """
        Log an audit trail entry.
        
        Args:
            action: The action being audited
            details: Details about the action
        """
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        
        with open(self.audit_file, "a") as f:
            json.dump(audit_entry, f)
            f.write("\n")
        
        self.logger.info(f"Audit: {action} | {json.dumps(details)}")
    
    def get_error_summary(self) -> Dict[str, int]:
        """
        Get a summary of errors by type and severity.
        
        Returns:
            Dictionary with error counts by type and severity
        """
        summary = {
            "total": len(self.errors),
            "by_type": {},
            "by_severity": {s.value: 0 for s in ErrorSeverity}
        }
        
        for error in self.errors:
            # Count by type
            if error.error_type not in summary["by_type"]:
                summary["by_type"][error.error_type] = 0
            summary["by_type"][error.error_type] += 1
            
            # Count by severity
            summary["by_severity"][error.severity.value] += 1
        
        return summary

# Global log manager instance
log_manager = LogManager()

def log_errors(func):
    """
    Decorator to automatically log errors from functions.
    
    Args:
        func: The function to wrap with error logging
    
    Returns:
        Wrapped function that logs errors
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_manager.log_error(e, {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            })
            raise
    return wrapper 