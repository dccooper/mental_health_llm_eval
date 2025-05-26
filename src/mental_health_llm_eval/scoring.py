"""
Scoring system for evaluating model responses.
Implements rubric-based scoring with validation and weighting.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from .exceptions import ValidationError

@dataclass(frozen=True)
class ScoringCriterion:
    """Individual scoring criterion."""
    name: str
    description: str
    weight: float
    min_score: float = 0.0
    max_score: float = 1.0
    red_flags: Optional[List[str]] = None

    def validate(self) -> None:
        """Validate criterion configuration."""
        if not 0 <= self.weight <= 1:
            raise ValidationError(f"Weight must be between 0 and 1 for {self.name}")
        
        if self.min_score >= self.max_score:
            raise ValidationError(f"min_score must be less than max_score for {self.name}")

@dataclass
class ScoringRubric:
    """Complete scoring rubric."""
    criteria: Dict[str, ScoringCriterion]
    
    def validate(self) -> None:
        """Validate rubric configuration."""
        if not self.criteria:
            raise ValidationError("Rubric must have at least one criterion")

        # Validate individual criteria
        for criterion in self.criteria.values():
            criterion.validate()

        # Validate total weight
        total_weight = sum(c.weight for c in self.criteria.values())
        if not 0.99 <= total_weight <= 1.01:
            raise ValidationError(f"Criterion weights must sum to 1.0 (got {total_weight})")

    def score_response(self, response: str, red_flags: List[str]) -> Dict[str, float]:
        """Score a response using this rubric."""
        scores = {}
        
        for name, criterion in self.criteria.items():
            # Check for critical red flags
            if criterion.red_flags and any(flag in red_flags for flag in criterion.red_flags):
                scores[name] = criterion.min_score
                continue
            
            # Calculate base score
            base_score = self._calculate_base_score(response, criterion)
            
            # Apply weight
            scores[name] = base_score * criterion.weight
        
        return scores

    def _calculate_base_score(self, response: str, criterion: ScoringCriterion) -> float:
        """Calculate the base score for a criterion."""
        # Implementation would go here - this is just a placeholder
        return criterion.max_score 