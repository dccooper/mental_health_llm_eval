"""
Tests for the logging and error handling module.
"""

import pytest
import json
import os
from pathlib import Path
from datetime import datetime
from mental_health_llm_eval.logging_handler import (
    LogManager,
    ErrorSeverity,
    EvalError,
    ValidationError,
    ModelError,
    SafetyError,
    SystemError,
    log_errors
)

@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary directory for logs."""
    log_dir = tmp_path / "test_logs"
    log_dir.mkdir()
    return str(log_dir)

@pytest.fixture
def log_manager(temp_log_dir):
    """Create a LogManager instance for testing."""
    return LogManager(log_dir=temp_log_dir)

def test_log_manager_initialization(temp_log_dir):
    """Test LogManager initialization and directory creation."""
    manager = LogManager(log_dir=temp_log_dir)
    
    # Check log directory exists
    assert Path(temp_log_dir).exists()
    assert Path(temp_log_dir).is_dir()
    
    # Check log files are created
    assert manager.log_file.exists()
    assert manager.audit_file.exists()

def test_error_logging(log_manager):
    """Test error logging functionality."""
    # Test basic error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_manager.log_error(e)
    
    # Check error was recorded
    assert len(log_manager.errors) == 1
    assert log_manager.errors[0].error_type == "ValueError"
    assert log_manager.errors[0].message == "Test error"
    
    # Test custom error with severity
    try:
        raise SafetyError("Critical safety issue")
    except Exception as e:
        log_manager.log_error(e)
    
    # Check error was recorded with correct severity
    assert len(log_manager.errors) == 2
    assert log_manager.errors[1].severity == ErrorSeverity.CRITICAL

def test_audit_logging(log_manager):
    """Test audit trail logging."""
    # Log an audit entry
    action = "test_action"
    details = {"key": "value"}
    log_manager.log_audit(action, details)
    
    # Read audit file and check entry
    with open(log_manager.audit_file) as f:
        audit_entry = json.loads(f.readline())
    
    assert audit_entry["action"] == action
    assert audit_entry["details"] == details
    assert "timestamp" in audit_entry

def test_error_decorator(log_manager):
    """Test the error logging decorator."""
    @log_errors
    def failing_function():
        raise ValueError("Decorator test error")
    
    # Call function and check error is logged
    with pytest.raises(ValueError):
        failing_function()
    
    assert len(log_manager.errors) == 1
    assert log_manager.errors[0].error_type == "ValueError"
    assert "failing_function" in str(log_manager.errors[0].additional_info)

def test_error_summary(log_manager):
    """Test error summary generation."""
    # Log various types of errors
    errors = [
        ValidationError("Invalid input"),
        ModelError("API error"),
        SafetyError("Safety violation"),
        SystemError("System issue")
    ]
    
    for error in errors:
        log_manager.log_error(error)
    
    # Get and check summary
    summary = log_manager.get_error_summary()
    
    assert summary["total"] == 4
    assert len(summary["by_type"]) == 4
    assert summary["by_severity"]["critical"] == 1  # SafetyError
    assert summary["by_severity"]["medium"] == 3    # Other errors

def test_thread_safety(log_manager):
    """Test thread-safe error logging."""
    import threading
    import time
    
    def log_errors_thread():
        for i in range(10):
            try:
                raise ValueError(f"Thread error {i}")
            except Exception as e:
                log_manager.log_error(e)
                time.sleep(0.01)  # Simulate work
    
    # Create and run multiple threads
    threads = [
        threading.Thread(target=log_errors_thread)
        for _ in range(3)
    ]
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Check all errors were logged
    assert len(log_manager.errors) == 30  # 3 threads * 10 errors each

def test_custom_error_classes():
    """Test custom error class functionality."""
    # Test ValidationError
    error = ValidationError("Invalid input")
    assert error.severity == ErrorSeverity.MEDIUM
    assert str(error) == "Invalid input"
    
    # Test SafetyError (should always be critical)
    error = SafetyError("Safety issue")
    assert error.severity == ErrorSeverity.CRITICAL
    
    # Test error timestamp
    error = ModelError("API error")
    assert isinstance(error.timestamp, datetime)

def test_log_file_content(log_manager):
    """Test log file content and format."""
    test_message = "Test log message"
    log_manager.logger.info(test_message)
    
    # Read log file
    with open(log_manager.log_file) as f:
        log_content = f.read()
    
    # Check log format
    assert test_message in log_content
    assert "[INFO]" in log_content
    assert "mental_health_eval" in log_content 