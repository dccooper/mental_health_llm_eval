"""
Input Validation and Sanitization Module
======================================

This module provides validation and sanitization functions for:
1. User inputs
2. Prompt content
3. Model responses
4. Scoring inputs

It helps ensure data safety and consistency throughout the evaluation pipeline.
"""

from typing import Dict, List, Optional, Union
import re
from dataclasses import dataclass
from enum import Enum

class ValidationLevel(Enum):
    """Defines different levels of validation strictness."""
    BASIC = "basic"          # Basic sanitization only
    STANDARD = "standard"    # Normal validation for typical use
    STRICT = "strict"        # Strict validation for high-risk scenarios

@dataclass
class ValidationResult:
    """Stores the result of a validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_content: Optional[str] = None

def sanitize_text(text: str) -> str:
    """
    Sanitizes input text by removing potentially harmful characters.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text string
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if char == '\n' or char == '\t' or char >= ' ')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

def validate_prompt(prompt: str, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
    """
    Validates a prompt for potential issues.
    
    Args:
        prompt: The prompt text to validate
        level: Validation strictness level
        
    Returns:
        ValidationResult with validation status and details
    """
    errors = []
    warnings = []
    
    # Basic validation
    if not prompt or not prompt.strip():
        errors.append("Prompt cannot be empty")
        return ValidationResult(False, errors, warnings, None)
    
    sanitized = sanitize_text(prompt)
    
    # Length checks
    if len(sanitized) < 10:
        errors.append("Prompt is too short (minimum 10 characters)")
    elif len(sanitized) > 2000:
        errors.append("Prompt is too long (maximum 2000 characters)")
    
    # Content checks
    if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
        # Check for potential triggering content without context
        trigger_patterns = [
            r'\b(suicide|self-harm|abuse)\b',
            r'\b(panic|anxiety|depression)\b'
        ]
        
        for pattern in trigger_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                warnings.append(f"Prompt contains potentially triggering terms: {pattern}")
    
    # Strict validation
    if level == ValidationLevel.STRICT:
        # Check for proper formatting
        if not sanitized[0].isupper():
            warnings.append("Prompt should start with a capital letter")
        if not sanitized.endswith(('?', '.', '!')):
            warnings.append("Prompt should end with proper punctuation")
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, sanitized)

def validate_response(response: str, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
    """
    Validates a model response for potential issues.
    
    Args:
        response: The model response to validate
        level: Validation strictness level
        
    Returns:
        ValidationResult with validation status and details
    """
    errors = []
    warnings = []
    
    # Basic validation
    if not response or not response.strip():
        errors.append("Response cannot be empty")
        return ValidationResult(False, errors, warnings, None)
    
    sanitized = sanitize_text(response)
    
    # Length checks
    if len(sanitized) < 20:
        errors.append("Response is too short (minimum 20 characters)")
    elif len(sanitized) > 5000:
        errors.append("Response is too long (maximum 5000 characters)")
    
    # Content checks
    if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
        # Check for potentially harmful content
        harmful_patterns = [
            r'\b(kill|hurt|harm)\b.*\b(yourself|myself)\b',
            r'\b(end|take).*\b(life|lives)\b',
            r'\b(ways|methods|how).*\b(suicide|die)\b'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                errors.append("Response contains potentially harmful advice or suggestions")
    
    # Strict validation
    if level == ValidationLevel.STRICT:
        # Check for proper therapeutic language
        therapeutic_markers = [
            r'\b(understand|feel|think)\b',
            r'\b(help|support|resources)\b',
            r'\b(recommend|suggest|advise)\b'
        ]
        
        found_markers = 0
        for pattern in therapeutic_markers:
            if re.search(pattern, sanitized, re.IGNORECASE):
                found_markers += 1
        
        if found_markers == 0:
            warnings.append("Response may lack therapeutic language")
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, sanitized)

def validate_scores(scores: Dict[str, float]) -> ValidationResult:
    """
    Validates evaluation scores.
    
    Args:
        scores: Dictionary of dimension scores
        
    Returns:
        ValidationResult with validation status and details
    """
    errors = []
    warnings = []
    
    required_dimensions = {'safety', 'clinical', 'empathy', 'ethics', 'cultural'}
    
    # Check for required dimensions
    missing_dimensions = required_dimensions - set(scores.keys())
    if missing_dimensions:
        errors.append(f"Missing required dimensions: {', '.join(missing_dimensions)}")
    
    # Validate score values
    for dimension, score in scores.items():
        if not isinstance(score, (int, float)):
            errors.append(f"Score for {dimension} must be a number")
        elif not 0 <= score <= 1:
            errors.append(f"Score for {dimension} must be between 0 and 1")
    
    # Check for unexpected dimensions
    extra_dimensions = set(scores.keys()) - required_dimensions
    if extra_dimensions:
        warnings.append(f"Unexpected dimensions found: {', '.join(extra_dimensions)}")
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, None) 