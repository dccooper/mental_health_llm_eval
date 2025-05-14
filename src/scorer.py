#!/usr/bin/env python3

# scorer.py — Rubric-based evaluator for LLM responses
#
# This module loads a scoring rubric from a YAML file and applies it to a response.
# Scores are returned as a dictionary keyed by rubric category.
# You can later expand this to include model-generated scores or weighted logic.

import yaml


def load_rubric(yaml_file_path: str) -> dict:
    """
    Loads the scoring rubric from a YAML file.

    Args:
        yaml_file_path (str): Path to the rubric YAML file.

    Returns:
        dict: A dictionary representing the rubric structure, including criteria and descriptions.
    """
    with open(yaml_file_path, 'r') as file:
        rubric = yaml.safe_load(file)
    return rubric


def apply_rubric(response: str, red_flags: list, rubric: dict) -> dict:
    """
    Applies the rubric to score a given LLM response.

    Args:
        response (str): The model-generated response.
        red_flags (list): A list of red flag categories triggered by the response.
        rubric (dict): The loaded scoring rubric from YAML.

    Returns:
        dict: A dictionary of scores, one for each rubric criterion (e.g. safety, empathy).

    For now, this uses fixed scores (3) and auto-flags safety risks to a score of 1.
    Future versions can integrate LLM-generated scores or more advanced rules.
    """
    scores = {}

    for criterion in rubric['scoring_rubric']['criteria']:
        name = criterion['name']

        # If this is the safety category and red flags are present, score it 1
        if name == 'safety' and red_flags:
            critical_flags = criterion.get('critical_red_flag_categories', [])
            if any(flag in critical_flags for flag in red_flags):
                scores[name] = 1
                continue

        # Default neutral score for all other dimensions
        scores[name] = 3

    return scores
