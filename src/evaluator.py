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
from src.models import generate_justification
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
    LogManager,
    log_errors,
    ErrorSeverity
)
from src.model_backends import (
    ModelBackend,
    ModelBackendType,
    create_backend,
    ModelResponse
)

@dataclass
class EvaluationResult:
    """Result of evaluating a single prompt."""
    prompt: str
    response: str
    red_flags: List[str]
    scores: Dict[str, float]
    justification: str
    metadata: Dict[str, Any]
    error: Optional[str] = None

class Evaluator:
    """Main evaluator class."""
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        rate_limits: Optional[Dict[str, Dict[str, int]]] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model_config: Model backend configuration with:
                - backend_type: Type of model backend
                - config: Backend-specific configuration
            validation_level: Level of input/output validation
            rate_limits: Custom rate limits for different operations
            log_dir: Directory for logs
        """
        # Initialize model backend
        backend_type = ModelBackendType(model_config["backend_type"])
        self.model = create_backend(backend_type, model_config["config"])
        
        # Set validation level
        self.validation_level = validation_level
        
        # Initialize rate limiters
        default_limits = {
            "prompts": {"rate": 100, "burst": 10},
            "model_calls": {"rate": 600, "burst": 5},
            "validation": {"rate": 1000, "burst": 20}
        }
        limits = rate_limits or default_limits
        
        self.rate_limiters = {
            LimitType.PROMPT: RateLimiter(**limits["prompts"]),
            LimitType.MODEL: RateLimiter(**limits["model_calls"]),
            LimitType.VALIDATION: RateLimiter(**limits["validation"])
        }
        
        # Initialize logging
        self.log_manager = LogManager(log_dir) if log_dir else None
    
    def evaluate_prompt(self, prompt: str) -> EvaluationResult:
        """
        Evaluate a single prompt.
        
        Args:
            prompt: The prompt to evaluate
            
        Returns:
            Evaluation result with response, scores, and metadata
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
            ValueError: If validation fails
        """
        try:
            # Check prompt rate limit
            self.rate_limiters[LimitType.PROMPT].check()
            
            # Validate prompt
            self.rate_limiters[LimitType.VALIDATION].check()
            validate_prompt(prompt, self.validation_level)
            
            # Get model response
            self.rate_limiters[LimitType.MODEL].check()
            response = self.model.query(prompt)
            
            if response.error:
                raise RuntimeError(f"Model error: {response.error}")
            
            # Validate response
            self.rate_limiters[LimitType.VALIDATION].check()
            validate_response(response.text, self.validation_level)
            
            # Check for red flags
            red_flags = check_red_flags(response.text)
            
            # Score response
            scores = score_response(prompt, response.text)
            
            # Validate scores
            self.rate_limiters[LimitType.VALIDATION].check()
            validate_scores(scores, self.validation_level)
            
            # Generate justification
            justification = generate_justification(prompt, response.text, scores)
            
            return EvaluationResult(
                prompt=prompt,
                response=response.text,
                red_flags=red_flags,
                scores=scores,
                justification=justification,
                metadata=response.metadata
            )
            
        except Exception as e:
            if self.log_manager:
                self.log_manager.log_error(
                    str(e),
                    severity=ErrorSeverity.ERROR,
                    context={
                        "prompt": prompt,
                        "validation_level": self.validation_level.name
                    }
                )
            return EvaluationResult(
                prompt=prompt,
                response="",
                red_flags=[],
                scores={},
                justification="",
                metadata={},
                error=str(e)
            )
    
    def evaluate_prompts(self, prompts: List[str]) -> List[EvaluationResult]:
        """
        Evaluate multiple prompts.
        
        Args:
            prompts: List of prompts to evaluate
            
        Returns:
            List of evaluation results
        """
        return [self.evaluate_prompt(prompt) for prompt in prompts]
    
    def load_prompts(self, yaml_path: str) -> List[str]:
        """
        Load prompts from YAML file.
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            List of prompts
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('prompts', [])
