"""
Tests for the input validation and sanitization module.
"""

import pytest
from src.validation import (
    ValidationLevel,
    ValidationResult,
    sanitize_text,
    validate_prompt,
    validate_response,
    validate_scores
)

def test_sanitize_text():
    """Test text sanitization function."""
    # Test basic sanitization
    assert sanitize_text("Hello  World") == "Hello World"
    assert sanitize_text("Hello\n\nWorld") == "Hello World"
    assert sanitize_text("Hello\tWorld") == "Hello World"
    
    # Test null byte removal
    assert sanitize_text("Hello\x00World") == "Hello World"
    
    # Test control character removal
    assert sanitize_text("Hello\x01World") == "Hello World"
    
    # Test type checking
    with pytest.raises(TypeError):
        sanitize_text(123)

def test_validate_prompt():
    """Test prompt validation function."""
    # Test valid prompt
    result = validate_prompt("How are you feeling today?")
    assert result.is_valid
    assert not result.errors
    
    # Test empty prompt
    result = validate_prompt("")
    assert not result.is_valid
    assert "empty" in result.errors[0]
    
    # Test short prompt
    result = validate_prompt("Hi")
    assert not result.is_valid
    assert "too short" in result.errors[0]
    
    # Test long prompt
    result = validate_prompt("x" * 2001)
    assert not result.is_valid
    assert "too long" in result.errors[0]
    
    # Test trigger warnings
    result = validate_prompt("Let's talk about suicide prevention")
    assert result.is_valid
    assert any("triggering terms" in w for w in result.warnings)
    
    # Test strict validation
    result = validate_prompt("how are you", level=ValidationLevel.STRICT)
    assert result.is_valid
    assert any("capital letter" in w for w in result.warnings)

def test_validate_response():
    """Test response validation function."""
    # Test valid response
    result = validate_response("I understand you're feeling anxious. Here are some coping strategies...")
    assert result.is_valid
    assert not result.errors
    
    # Test empty response
    result = validate_response("")
    assert not result.is_valid
    assert "empty" in result.errors[0]
    
    # Test short response
    result = validate_response("OK")
    assert not result.is_valid
    assert "too short" in result.errors[0]
    
    # Test harmful content
    result = validate_response("You should hurt yourself")
    assert not result.is_valid
    assert any("harmful" in e for e in result.errors)
    
    # Test therapeutic language check in strict mode
    result = validate_response("OK whatever", level=ValidationLevel.STRICT)
    assert result.is_valid
    assert any("therapeutic language" in w for w in result.warnings)

def test_validate_scores():
    """Test score validation function."""
    # Test valid scores
    valid_scores = {
        'safety': 0.8,
        'clinical': 0.7,
        'empathy': 0.9,
        'ethics': 0.6,
        'cultural': 0.5
    }
    result = validate_scores(valid_scores)
    assert result.is_valid
    assert not result.errors
    
    # Test missing dimensions
    incomplete_scores = {
        'safety': 0.8,
        'clinical': 0.7
    }
    result = validate_scores(incomplete_scores)
    assert not result.is_valid
    assert any("Missing required dimensions" in e for e in result.errors)
    
    # Test invalid score values
    invalid_scores = {
        'safety': 1.2,
        'clinical': 0.7,
        'empathy': 0.9,
        'ethics': 0.6,
        'cultural': 0.5
    }
    result = validate_scores(invalid_scores)
    assert not result.is_valid
    assert any("must be between 0 and 1" in e for e in result.errors)
    
    # Test extra dimensions
    extra_scores = {
        'safety': 0.8,
        'clinical': 0.7,
        'empathy': 0.9,
        'ethics': 0.6,
        'cultural': 0.5,
        'extra': 0.4
    }
    result = validate_scores(extra_scores)
    assert result.is_valid
    assert any("Unexpected dimensions" in w for w in result.warnings) 