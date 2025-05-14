#!/usr/bin/env python3

"""
Core Evaluator Module
===================

This module connects the full evaluation pipeline:
1. Load prompts from YAML
2. Get model response for each prompt
3. Check for red flags
4. Score the response using rubric and criteria
5. Generate a justification
6. Return all results as structured data
"""

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
from src.rate_limiter import (
    RateLimiter,
    LimitType,
    RateLimitExceeded
)
from src.logging_handler import (
    log_manager,
    log_errors,
    ValidationError,
    ModelError,
    SafetyError,
    SystemError
)


class EvaluationError(Exception):
    """Raised when evaluation encounters a validation error."""
    pass


@dataclass
class PromptEntry:
    """Structure for a single prompt entry."""
    id: str
    prompt: str
    category: str
    subcategory: str
    context: Optional[str]
    expected_behaviors: List[str]
    scoring_criteria: Dict[str, List[Dict[str, float]]]


@dataclass
class EvaluationResult:
    """Structure for evaluation results."""
    prompt_entry: PromptEntry
    model_response: str
    detected_red_flags: List[str]
    scores: Dict[str, float]
    dimension_scores: Dict[str, Dict[str, float]]
    justification: str
    meets_expected_behaviors: List[bool]
    total_score: float
    validation_warnings: List[str]


class Evaluator:
    """
    Main evaluator class that handles rate limiting and evaluation pipeline.
    """
    
    def __init__(self):
        """Initialize evaluator with rate limiter."""
        self.rate_limiter = RateLimiter()
    
    @log_errors
    def validate_prompt_entry(
        self,
        entry: PromptEntry,
        level: ValidationLevel = ValidationLevel.STANDARD
    ) -> List[str]:
        """
        Validates a prompt entry and returns any warnings.
        
        Args:
            entry: The prompt entry to validate
            level: Validation strictness level
            
        Returns:
            List of validation warnings
            
        Raises:
            ValidationError: If the prompt entry fails validation
        """
        try:
            # Check validation rate limit
            self.rate_limiter.try_acquire(LimitType.VALIDATION)
            
            # Validate prompt text
            prompt_validation = validate_prompt(entry.prompt, level)
            if not prompt_validation.is_valid:
                raise ValidationError(f"Invalid prompt: {'; '.join(prompt_validation.errors)}")
            
            # Validate scoring criteria
            if not entry.scoring_criteria:
                raise ValidationError("Missing scoring criteria")
            
            warnings = prompt_validation.warnings
            
            # Additional prompt entry validations
            if not entry.id:
                raise ValidationError("Missing prompt ID")
            if not entry.category:
                raise ValidationError("Missing category")
            if not entry.subcategory:
                warnings.append("Missing subcategory")
            
            # Log successful validation
            log_manager.log_audit("prompt_validation", {
                "prompt_id": entry.id,
                "level": level.value,
                "warnings": len(warnings)
            })
            
            return warnings
            
        except RateLimitExceeded as e:
            raise ValidationError(f"Validation rate limit exceeded: {str(e)}")
    
    @log_errors
    def evaluate_response(
        self,
        prompt_entry: PromptEntry,
        model_response: str,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> EvaluationResult:
        """
        Evaluates a single model response against the prompt's criteria.

        Args:
            prompt_entry: The structured prompt entry
            model_response: The response from the model
            validation_level: How strict the validation should be

        Returns:
            Complete evaluation results
            
        Raises:
            ValidationError: If validation fails
            ModelError: If model interaction fails
            SafetyError: If safety checks fail
            RateLimitExceeded: If rate limits are exceeded
        """
        try:
            # Check prompt evaluation rate limit
            self.rate_limiter.try_acquire(LimitType.PROMPT)
            
            # Start evaluation audit trail
            log_manager.log_audit("evaluation_start", {
                "prompt_id": prompt_entry.id,
                "validation_level": validation_level.value
            })
            
            # Collect validation warnings
            validation_warnings = self.validate_prompt_entry(prompt_entry, validation_level)
            
            # Validate model response
            response_validation = validate_response(model_response, validation_level)
            if not response_validation.is_valid:
                raise ValidationError(f"Invalid response: {'; '.join(response_validation.errors)}")
            validation_warnings.extend(response_validation.warnings)
            
            # Use sanitized response for evaluation
            sanitized_response = response_validation.sanitized_content
            
            # Check for red flags in response
            detected_flags = check_red_flags(sanitized_response)
            if detected_flags:
                log_manager.log_error(
                    SafetyError(f"Red flags detected: {', '.join(detected_flags)}"),
                    {"prompt_id": prompt_entry.id, "flags": detected_flags}
                )
            
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
                raise ValidationError(f"Invalid scores: {'; '.join(score_validation.errors)}")
            validation_warnings.extend(score_validation.warnings)
            
            # Check if response meets expected behaviors
            meets_behaviors = [
                any(behavior.lower() in sanitized_response.lower() 
                    for behavior in prompt_entry.expected_behaviors)
            ]
            
            # Check model call rate limit before generating justification
            self.rate_limiter.try_acquire(LimitType.MODEL_CALL)
            
            try:
                # Generate explanation using LLM
                justification = generate_justification(
                    prompt=prompt_entry.prompt,
                    context=prompt_entry.context,
                    response=sanitized_response,
                    scores=scores,
                    red_flags=detected_flags
                )
            except Exception as e:
                log_manager.log_error(
                    ModelError(f"Failed to generate justification: {str(e)}"),
                    {"prompt_id": prompt_entry.id}
                )
                justification = "[Justification generation failed]"
            
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
            
            result = EvaluationResult(
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
            
            # Log successful evaluation
            log_manager.log_audit("evaluation_complete", {
                "prompt_id": prompt_entry.id,
                "total_score": total_score,
                "red_flags": len(detected_flags),
                "warnings": len(validation_warnings)
            })
            
            return result
            
        except Exception as e:
            # Log failure and re-raise
            log_manager.log_error(e, {
                "prompt_id": prompt_entry.id,
                "stage": "evaluation"
            })
            raise
    
    @log_errors
    def run_evaluation(
        self,
        prompt_file_path: str,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> Tuple[List[EvaluationResult], List[Dict[str, Any]]]:
        """
        Runs the full evaluation loop over a prompt file.

        Args:
            prompt_file_path: Path to YAML prompt file
            validation_level: How strict the validation should be

        Returns:
            Tuple containing:
            - List of successful evaluation results
            - List of failed evaluations with error details
        """
        try:
            log_manager.log_audit("batch_evaluation_start", {
                "prompt_file": prompt_file_path,
                "validation_level": validation_level.value
            })
            
            prompts = load_prompts(prompt_file_path)
            results = []
            failures = []

            for prompt_entry in prompts:
                try:
                    # Check model call rate limit
                    self.rate_limiter.try_acquire(LimitType.MODEL_CALL)
                    
                    model_output = query_model(prompt_entry.prompt)
                    evaluation = self.evaluate_response(
                        prompt_entry,
                        model_output,
                        validation_level
                    )
                    results.append(evaluation)
                    
                except RateLimitExceeded as e:
                    failures.append({
                        'prompt_id': prompt_entry.id,
                        'error': f"Rate limit exceeded: {str(e)}",
                        'category': prompt_entry.category
                    })
                except (ValidationError, ModelError, SafetyError) as e:
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

            log_manager.log_audit("batch_evaluation_complete", {
                "total_prompts": len(prompts),
                "successful": len(results),
                "failed": len(failures)
            })

            return results, failures
            
        except Exception as e:
            log_manager.log_error(
                SystemError(f"Batch evaluation failed: {str(e)}"),
                {"prompt_file": prompt_file_path}
            )
            raise
    
    def get_rate_limit_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current rate limit status for all operation types.
        
        Returns:
            Dictionary with rate limit status for each operation type
        """
        return {
            limit_type.value: self.rate_limiter.get_status(limit_type)
            for limit_type in LimitType
        }


# Module-level evaluator instance for backward compatibility
_default_evaluator = Evaluator()

def evaluate_response(*args, **kwargs) -> EvaluationResult:
    """Backward compatible wrapper for evaluate_response."""
    return _default_evaluator.evaluate_response(*args, **kwargs)

def run_evaluation(*args, **kwargs) -> Tuple[List[EvaluationResult], List[Dict[str, Any]]]:
    """Backward compatible wrapper for run_evaluation."""
    return _default_evaluator.run_evaluation(*args, **kwargs)

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
