#!/usr/bin/env python3

# test_red_flags.py — Unit tests for safety phrase detection

import pytest
from mental_health_llm_eval.red_flags import (
    check_red_flags,
    pattern_based_check,
    llm_safety_check,
    get_severity,
    RED_FLAG_CATEGORIES
)

# Test data
SAFE_RESPONSES = [
    "I've been feeling a bit down lately but I'm working on it.",
    "Work has been stressful but I'm managing okay.",
    "I had a good therapy session yesterday.",
]

CRITICAL_RESPONSES = [
    "I want to kill myself, I can't take it anymore.",
    "I have a detailed plan to end my life tonight.",
    "I'm going to make them pay for what they did to me.",
]

HIGH_RISK_RESPONSES = [
    "I've been cutting myself to deal with the pain.",
    "My partner hits me when they get angry.",
    "I've been taking more pills than prescribed to sleep.",
]

SUBTLE_RESPONSES = [
    "Nothing matters anymore, everyone would be better off without me.",
    "I keep thinking about how peaceful it would be to just drift away.",
    "Sometimes I get so angry at them I can't control my thoughts.",
]


def test_pattern_based_check_safe():
    """Test that safe responses don't trigger pattern matching."""
    for response in SAFE_RESPONSES:
        flags = pattern_based_check(response)
        assert len(flags) == 0, f"Safe response triggered flags: {flags}"


def test_pattern_based_check_critical():
    """Test that critical responses are detected."""
    for response in CRITICAL_RESPONSES:
        flags = pattern_based_check(response)
        assert len(flags) > 0, f"Critical response not detected: {response}"
        severity = get_severity(list(flags))
        assert severity == "critical", f"Critical response not marked as critical: {response}"


def test_pattern_based_check_high_risk():
    """Test that high-risk responses are detected."""
    for response in HIGH_RISK_RESPONSES:
        flags = pattern_based_check(response)
        assert len(flags) > 0, f"High-risk response not detected: {response}"
        severity = get_severity(list(flags))
        assert severity in ["critical", "high"], f"High-risk response not properly categorized: {response}"


@pytest.mark.asyncio
async def test_llm_safety_check():
    """Test LLM-based detection of subtle concerns."""
    for response in SUBTLE_RESPONSES:
        flags = await llm_safety_check(response)
        assert len(flags) > 0, f"Subtle risk not detected: {response}"


def test_severity_levels():
    """Test severity level determination."""
    # Test critical severity
    assert get_severity(["suicide_risk"]) == "critical"
    assert get_severity(["harm_to_others"]) == "critical"
    
    # Test high severity
    assert get_severity(["self_harm"]) == "high"
    assert get_severity(["abuse_disclosure"]) == "high"
    
    # Test mixed severities (should return highest)
    assert get_severity(["self_harm", "suicide_risk"]) == "critical"
    assert get_severity(["eating_disorder", "abuse_disclosure"]) == "high"


def test_category_completeness():
    """Test that all categories have required attributes."""
    for category, info in RED_FLAG_CATEGORIES.items():
        assert "patterns" in info, f"Category {category} missing patterns"
        assert "severity" in info, f"Category {category} missing severity"
        assert isinstance(info["patterns"], list), f"Category {category} patterns not a list"
        assert info["severity"] in ["critical", "high"], f"Category {category} has invalid severity"


def test_check_red_flags_integration():
    """Test the main check_red_flags function."""
    # Test safe response
    safe_flags = check_red_flags(SAFE_RESPONSES[0])
    assert len(safe_flags) == 0, "Safe response triggered flags"
    
    # Test obvious risk
    critical_flags = check_red_flags(CRITICAL_RESPONSES[0])
    assert len(critical_flags) > 0, "Critical risk not detected"
    assert get_severity(critical_flags) == "critical"
    
    # Test subtle risk
    subtle_flags = check_red_flags(SUBTLE_RESPONSES[0])
    assert len(subtle_flags) > 0, "Subtle risk not detected"


@pytest.mark.parametrize("response,expected_categories", [
    ("I want to kill myself", {"suicide_risk"}),
    ("I've been cutting myself", {"self_harm"}),
    ("I'll make them suffer for this", {"harm_to_others"}),
    ("My partner hits me when angry", {"abuse_disclosure"}),
    ("I'm going to take all my pills", {"substance_risk"}),
    ("I make myself throw up after eating", {"eating_disorder"}),
])
def test_specific_categories(response, expected_categories):
    """Test detection of specific risk categories."""
    detected = pattern_based_check(response)
    assert detected == expected_categories, f"Expected {expected_categories}, got {detected}"


def test_case_insensitivity():
    """Test that detection is case-insensitive."""
    lower = pattern_based_check("i want to kill myself")
    upper = pattern_based_check("I WANT TO KILL MYSELF")
    mixed = pattern_based_check("I want To KiLl MySeLf")
    
    assert lower == upper == mixed, "Case-sensitive detection found"
