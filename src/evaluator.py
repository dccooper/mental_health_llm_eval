#!/usr/bin/env python3

# evaluator.py — Core logic for running LLM evaluations
#
# This module connects the full evaluation pipeline:
# 1. Load prompts from YAML
# 2. Get model response for each prompt
# 3. Check for red flags
# 4. Score the response using rubric
# 5. Generate a justification (stubbed or real)
# 6. Return all results as structured data

import yaml
from src.models import query_model, generate_justification
from src.red_flags import check_red_flags
from src.scorer import load_rubric, apply_rubric


def load_prompts(yaml_file_path: str) -> list:
    """
    Loads a YAML prompt bank and returns a flattened list of prompt entries.

    Args:
        yaml_file_path (str): Path to the YAML file containing prompt definitions.

    Returns:
        list: A list of prompts with category and subcategory metadata included.

    Each prompt entry is a dict with:
        - category
        - subcategory
        - prompt (text)
    """
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    prompts = []
    for category, items in data['prompt_bank'].items():
        for item in items:
            prompts.append({
                "category": category,
                "subcategory": item.get('subcategory', ''),
                "prompt": item['prompt']
            })

    return prompts


def evaluate_response(prompt: str, model_response: str, rubric: dict) -> dict:
    """
    Evaluates a single model response against the rubric and red flag rules.

    Args:
        prompt (str): The original prompt.
        model_response (str): The response from the model.
        rubric (dict): The scoring rubric loaded from YAML.

    Returns:
        dict: A full evaluation record, including:
            - prompt
            - response
            - red_flags
            - scores
            - justification
    """
    red_flags = check_red_flags(model_response)
    scores = apply_rubric(model_response, red_flags, rubric)
    justification = generate_justification(prompt, model_response, scores)

    return {
        "prompt": prompt,
        "response": model_response,
        "red_flags": red_flags,
        "scores": scores,
        "justification": justification
    }


def run_evaluation(prompt_file_path: str, rubric_file_path: str) -> list:
    """
    Runs the full evaluation loop over a prompt file and rubric.

    Args:
        prompt_file_path (str): Path to YAML prompt file.
        rubric_file_path (str): Path to YAML scoring rubric.

    Returns:
        list: List of evaluations (one per prompt), each containing structured metadata.
    """
    prompts = load_prompts(prompt_file_path)
    rubric = load_rubric(rubric_file_path)
    results = []

    for prompt_entry in prompts:
        model_output = query_model(prompt_entry['prompt'])
        evaluation = evaluate_response(prompt_entry['prompt'], model_output, rubric)
        evaluation['category'] = prompt_entry['category']
        evaluation['subcategory'] = prompt_entry['subcategory']
        results.append(evaluation)

    return results
