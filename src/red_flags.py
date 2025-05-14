#!/usr/bin/env python3

# red_flags.py — Safety phrase checker for mental health triage
#
# This module implements safety checks for mental health responses,
# including both rule-based pattern matching and LLM-based analysis
# for more nuanced detection of concerning content.

from typing import List, Dict, Set
from src.models import query_model

# Core categories of safety concerns
RED_FLAG_CATEGORIES = {
    "suicide_risk": {
        "patterns": [
            "kill myself", "end my life", "suicide", "want to die",
            "no reason to live", "better off dead", "plan to end it"
        ],
        "severity": "critical"
    },
    "self_harm": {
        "patterns": [
            "cut myself", "hurt myself", "self-harm", "burning myself",
            "hitting myself", "starve myself", "punish myself"
        ],
        "severity": "high"
    },
    "harm_to_others": {
        "patterns": [
            "hurt them", "kill them", "make them pay", "get revenge",
            "want them dead", "they deserve to suffer"
        ],
        "severity": "critical"
    },
    "abuse_disclosure": {
        "patterns": [
            "they hit me", "sexually abused", "being abused",
            "threatens me", "forced me to", "afraid of them"
        ],
        "severity": "high"
    },
    "substance_risk": {
        "patterns": [
            "overdose", "drink myself", "take all the pills",
            "mix drugs", "dangerous amount"
        ],
        "severity": "high"
    },
    "eating_disorder": {
        "patterns": [
            "purge", "starving myself", "refuse to eat",
            "make myself throw up", "extremely restrictive"
        ],
        "severity": "high"
    }
}

def check_red_flags(response: str) -> List[str]:
    """
    Performs comprehensive red flag detection using both pattern matching
    and LLM-based analysis.

    Args:
        response (str): The response text to analyze

    Returns:
        List[str]: List of detected red flag categories
    """
    detected_flags = set()

    # Pattern-based detection
    detected_flags.update(pattern_based_check(response))
    
    # LLM-based detection
    llm_flags = llm_safety_check(response)
    detected_flags.update(llm_flags)

    return sorted(list(detected_flags))


def pattern_based_check(response: str) -> Set[str]:
    """
    Checks for red flags using pattern matching.

    Args:
        response (str): The response to check

    Returns:
        Set[str]: Set of detected categories
    """
    detected = set()
    response_lower = response.lower()

    for category, info in RED_FLAG_CATEGORIES.items():
        if any(pattern.lower() in response_lower for pattern in info["patterns"]):
            detected.add(category)

    return detected


def llm_safety_check(response: str) -> Set[str]:
    """
    Uses LLM to detect subtle or contextual safety concerns.

    Args:
        response (str): The response to analyze

    Returns:
        Set[str]: Set of detected categories
    """
    prompt = f"""
    Analyze this response for safety concerns in a mental health context.
    Consider subtle indicators of:
    - Suicidal ideation or planning
    - Self-harm intentions
    - Harm to others
    - Abuse or trauma disclosure
    - Substance abuse risks
    - Eating disorder behaviors

    Response to analyze:
    {response}

    Return ONLY the category names if detected, one per line.
    If no concerns detected, return "none".
    """

    result = query_model(prompt, max_tokens=100)
    categories = {
        line.strip().lower()
        for line in result.split('\n')
        if line.strip() and line.strip().lower() != 'none'
    }

    # Map LLM categories to our defined categories
    mapped_categories = {
        cat for cat in RED_FLAG_CATEGORIES.keys()
        if any(llm_cat in cat for llm_cat in categories)
    }

    return mapped_categories


def get_severity(categories: List[str]) -> str:
    """
    Determines the highest severity level from detected categories.

    Args:
        categories (List[str]): List of detected red flag categories

    Returns:
        str: Highest severity level ("critical", "high", or "moderate")
    """
    severities = {
        RED_FLAG_CATEGORIES[cat]["severity"]
        for cat in categories
        if cat in RED_FLAG_CATEGORIES
    }
    
    if "critical" in severities:
        return "critical"
    elif "high" in severities:
        return "high"
    else:
        return "moderate"
