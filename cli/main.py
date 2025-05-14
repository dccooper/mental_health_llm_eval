#!/usr/bin/env python3

# cli/main.py — Command-line batch evaluator for mental health LLM prompts

import argparse
import json
import pandas as pd
from src.evaluator import run_evaluation

def save_results(results, output_base: str):
    """
    Saves JSON and CSV results from the evaluation.
    """
    with open(f"{output_base}.json", "w") as jf:
        json.dump(results, jf, indent=2)

    df = pd.json_normalize(results)
    df.to_csv(f"{output_base}.csv", index=False)

def main():
    """
    CLI entry point. Expects:
    --prompts : path to prompt YAML
    --rubric  : path to rubric YAML
    --output  : base filename (no extension)
    """
    parser = argparse.ArgumentParser(description="Batch-evaluate LLM mental health responses.")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--output", default="eval_results")

    args = parser.parse_args()
    results = run_evaluation(args.prompts, args.rubric)
    save_results(results, args.output)
    print(f"✅ Results saved to {args.output}.json and .csv")
