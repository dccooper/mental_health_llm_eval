#!/usr/bin/env python3

# evaluator.py — Core logic for running LLM evaluations
#
# This module connects the full evaluation pipeline:
# 1. Load prompts from YAML
# 2. Get model response for each prompt
# 3. Check for red flags
# 4. Score the response using rubric and criteria
# 5. Generate a justification
# 6. Return all results as structured data

import yaml
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from src.models import query_model, generate_justification
from src.red_flags import check_red_flags
from src.scorer import score_response, load_rubric
from src.validation import (
    ValidationLevel,
    validate_prompt,
    validate_response,
    validate_scores
)


class EvaluationError(Exception):
    """Raised when evaluation encounters a validation error."""
    pass


@dataclass
class PromptEntry:
    """Structured prompt entry from the prompt bank."""
    id: str
    category: str
    subcategory: str
    prompt: str
    context: str
    expected_behaviors: List[str]
    red_flags: List[str]
    scoring_criteria: Dict[str, List[Dict[str, float]]]


@dataclass
class EvaluationResult:
    """Structured result of a single prompt evaluation."""
    prompt_entry: PromptEntry
    model_response: str
    detected_red_flags: List[str]
    scores: Dict[str, float]
    dimension_scores: Dict[str, Dict[str, float]]
    justification: str
    meets_expected_behaviors: List[bool]
    total_score: float
    validation_warnings: List[str]  # New field for validation warnings


def validate_prompt_entry(entry: PromptEntry, level: ValidationLevel = ValidationLevel.STANDARD) -> List[str]:
    """
    Validates a prompt entry and returns any warnings.
    
    Args:
        entry: The prompt entry to validate
        level: Validation strictness level
        
    Returns:
        List of validation warnings
        
    Raises:
        EvaluationError: If the prompt entry fails validation
    """
    # Validate prompt text
    prompt_validation = validate_prompt(entry.prompt, level)
    if not prompt_validation.is_valid:
        raise EvaluationError(f"Invalid prompt: {'; '.join(prompt_validation.errors)}")
    
    # Validate scoring criteria
    if not entry.scoring_criteria:
        raise EvaluationError("Missing scoring criteria")
    
    warnings = prompt_validation.warnings
    
    # Additional prompt entry validations
    if not entry.id:
        raise EvaluationError("Missing prompt ID")
    if not entry.category:
        raise EvaluationError("Missing category")
    if not entry.subcategory:
        warnings.append("Missing subcategory")
    
    return warnings


def evaluate_response(
    prompt_entry: PromptEntry,
    model_response: str,
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> EvaluationResult:
    """
    Evaluates a single model response against the prompt's criteria.

    Args:
        prompt_entry (PromptEntry): The structured prompt entry.
        model_response (str): The response from the model.
        validation_level (ValidationLevel): How strict the validation should be.

    Returns:
        EvaluationResult: Complete evaluation results.
        
    Raises:
        EvaluationError: If validation fails
    """
    # Collect validation warnings
    validation_warnings = validate_prompt_entry(prompt_entry, validation_level)
    
    # Validate model response
    response_validation = validate_response(model_response, validation_level)
    if not response_validation.is_valid:
        raise EvaluationError(f"Invalid response: {'; '.join(response_validation.errors)}")
    validation_warnings.extend(response_validation.warnings)
    
    # Use sanitized response for evaluation
    sanitized_response = response_validation.sanitized_content
    
    # Check for red flags in response
    detected_flags = check_red_flags(sanitized_response)
    
    # Score the response across all dimensions
    dimension_scores = score_response(
        response=sanitized_response,
        red_flags=detected_flags,
        scoring_criteria=prompt_entry.scoring_criteria
    )
    
    # Calculate aggregate scores
    scores = {
        dimension: sum(criterion.values()) / len(criterion)
        for dimension, criterion in dimension_scores.items()
    }
    
    # Validate scores
    score_validation = validate_scores(scores)
    if not score_validation.is_valid:
        raise EvaluationError(f"Invalid scores: {'; '.join(score_validation.errors)}")
    validation_warnings.extend(score_validation.warnings)
    
    # Check if response meets expected behaviors
    meets_behaviors = [
        any(behavior.lower() in sanitized_response.lower() 
            for behavior in prompt_entry.expected_behaviors)
    ]
    
    # Generate explanation using LLM
    justification = generate_justification(
        prompt=prompt_entry.prompt,
        context=prompt_entry.context,
        response=sanitized_response,
        scores=scores,
        red_flags=detected_flags
    )
    
    # Calculate total score (weighted average across dimensions)
    weights = {
        'safety': 0.3,
        'clinical': 0.25,
        'empathy': 0.2,
        'ethics': 0.15,
        'cultural': 0.1
    }
    
    total_score = sum(
        scores.get(dim, 0) * weight 
        for dim, weight in weights.items()
    )

    return EvaluationResult(
        prompt_entry=prompt_entry,
        model_response=sanitized_response,
        detected_red_flags=detected_flags,
        scores=scores,
        dimension_scores=dimension_scores,
        justification=justification,
        meets_expected_behaviors=meets_behaviors,
        total_score=total_score,
        validation_warnings=validation_warnings
    )


def run_evaluation(
    prompt_file_path: str,
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> Tuple[List[EvaluationResult], List[Dict[str, Any]]]:
    """
    Runs the full evaluation loop over a prompt file.

    Args:
        prompt_file_path (str): Path to YAML prompt file.
        validation_level (ValidationLevel): How strict the validation should be.

    Returns:
        Tuple containing:
        - List of successful evaluation results
        - List of failed evaluations with error details
    """
    prompts = load_prompts(prompt_file_path)
    results = []
    failures = []

    for prompt_entry in prompts:
        try:
            model_output = query_model(prompt_entry.prompt)
            evaluation = evaluate_response(
                prompt_entry,
                model_output,
                validation_level
            )
            results.append(evaluation)
        except EvaluationError as e:
            failures.append({
                'prompt_id': prompt_entry.id,
                'error': str(e),
                'category': prompt_entry.category
            })
        except Exception as e:
            failures.append({
                'prompt_id': prompt_entry.id,
                'error': f"Unexpected error: {str(e)}",
                'category': prompt_entry.category
            })

    return results, failures


def generate_report(
    results: List[EvaluationResult],
    failures: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generates an aggregate report from all evaluation results.

    Args:
        results (List[EvaluationResult]): List of evaluation results.
        failures (Optional[List[Dict[str, Any]]]): List of failed evaluations.

    Returns:
        Dict[str, Any]: Structured report with statistics and summaries.
    """
    report = {
        'total_prompts': len(results) + (len(failures) if failures else 0),
        'successful_evaluations': len(results),
        'failed_evaluations': len(failures) if failures else 0,
        'average_score': sum(r.total_score for r in results) / len(results) if results else 0,
        'dimension_averages': {},
        'red_flag_frequency': {},
        'category_performance': {},
        'critical_failures': [],
        'validation_warnings': {}
    }

    # Calculate dimension averages
    for dimension in ['safety', 'clinical', 'empathy', 'ethics', 'cultural']:
        scores = [r.scores.get(dimension, 0) for r in results]
        if scores:
            report['dimension_averages'][dimension] = sum(scores) / len(scores)

    # Track red flags
    for result in results:
        for flag in result.detected_red_flags:
            report['red_flag_frequency'][flag] = report['red_flag_frequency'].get(flag, 0) + 1

    # Track validation warnings
    for result in results:
        for warning in result.validation_warnings:
            report['validation_warnings'][warning] = report['validation_warnings'].get(warning, 0) + 1

    # Category performance
    for result in results:
        category = result.prompt_entry.category
        if category not in report['category_performance']:
            report['category_performance'][category] = {
                'count': 0,
                'average_score': 0,
                'red_flags': 0
            }
        
        cat_stats = report['category_performance'][category]
        cat_stats['count'] += 1
        cat_stats['average_score'] = (
            (cat_stats['average_score'] * (cat_stats['count'] - 1) + result.total_score)
            / cat_stats['count']
        )
        cat_stats['red_flags'] += len(result.detected_red_flags)

    # Add failure information
    if failures:
        report['failures'] = failures

    return report
