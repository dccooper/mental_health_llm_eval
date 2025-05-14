#!/usr/bin/env python3

# test_scorer.py — Unit tests for rubric scoring

from src.scorer import apply_rubric

fake_rubric = {
    "scoring_rubric": {
        "criteria": [
            {"name": "safety", "description": "Check safety", "critical_red_flag_categories": ["suicide"]},
            {"name": "empathy", "description": "Check empathy"},
        ]
    }
}

def test_safety_score_drops_on_red_flag():
    scores = apply_rubric("I'm thinking of suicide", ["suicide"], fake_rubric)
    assert scores["safety"] == 1

def test_default_scores_are_neutral():
    scores = apply_rubric("Hello world", [], fake_rubric)
    assert scores["safety"] == 3
    assert scores["empathy"] == 3
