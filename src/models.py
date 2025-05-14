#!/usr/bin/env python3

# models.py — Model interface layer
#
# This module defines how to interact with language models used for scoring and generating justifications.
# Initially, both functions return stub responses for testing. Later, you can connect them to OpenAI, Claude, or local models.

# Simulates an LLM response to the provided prompt.
def query_model(prompt: str, model_name: str = "gpt-4") -> str:
    """
    Simulates sending a prompt to a language model and returns a placeholder response.

    Args:
        prompt (str): The user's prompt (e.g. clinical scenario or question).
        model_name (str): The name of the model to simulate (e.g. "gpt-4").

    Returns:
        str: Stubbed model response for development and testing.

    Replace this function with a real model call for production.
    """
    return f"[Stub response from {model_name} to prompt: '{prompt}']"


# Simulates generating an explanation for the assigned scores.
def generate_justification(prompt: str, response: str, scores: dict, model_name: str = "gpt-4") -> str:
    """
    Simulates generating a justification for a scored LLM response.

    Args:
        prompt (str): The original user input.
        response (str): The model’s generated reply.
        scores (dict): A dictionary of evaluation scores for the response.
        model_name (str): The model used to generate justification (for logging or future use).

    Returns:
        str: Stub justification string.

    In future versions, this function can call an LLM to generate a paragraph explaining the score rationale.
    """
    return f"[LLM justification based on scores {scores} for response: '{response}']"
