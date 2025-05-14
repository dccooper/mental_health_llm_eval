#!/usr/bin/env python3
# __main__.py — Modular entry point for mental_health_llm_eval
#
# This file enables multi-mode execution of the toolkit using:
#   python -m mental_health_llm_eval [--cli | --ui | --validate | --version]
#
# It's intended as a convenience layer for common developer or evaluator workflows.

import sys
import argparse
import os
import subprocess
from src.cli import main as cli_main
from src.evaluator import load_prompts
from src.scorer import load_rubric

TOOL_VERSION = "0.1.0"

def launch_ui():
    """
    Launches the Streamlit-based UI app.
    """
    print("🔄 Launching Streamlit interface...")
    subprocess.run(["streamlit", "run", "app/streamlit_app.py"])


def validate_inputs(prompt_path, rubric_path):
    """
    Validates prompt bank and rubric YAML files.
    """
    try:
        load_prompts(prompt_path)
        print(f"✅ Prompts file at '{prompt_path}' loaded successfully.")
    except Exception as e:
        print(f"❌ Error in prompts file: {e}")
        return

    try:
        load_rubric(rubric_path)
        print(f"✅ Rubric file at '{rubric_path}' loaded successfully.")
    except Exception as e:
        print(f"❌ Error in rubric file: {e}")


def main():
    """
    Dispatches command-line options to the appropriate run mode.
    """
    parser = argparse.ArgumentParser(description="Mental Health LLM Eval Toolkit")
    parser.add_argument("--cli", action="store_true", help="Run CLI batch evaluator (default)")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit UI")
    parser.add_argument("--validate", action="store_true", help="Validate prompt and rubric files")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--prompts", help="Path to prompt_bank.yaml (used by --cli or --validate)")
    parser.add_argument("--rubric", help="Path to scoring_rubric.yaml (used by --cli or --validate)")

    args = parser.parse_args()

    if args.version:
        print(f"🧠 mental_health_llm_eval version {TOOL_VERSION}")
        return

    if args.ui:
        launch_ui()
        return

    if args.validate:
        if not args.prompts or not args.rubric:
            print("⚠️ Please provide both --prompts and --rubric to validate.")
            return
        validate_inputs(args.prompts, args.rubric)
        return

    # Default: run CLI evaluator
    if args.prompts and args.rubric:
        cli_main.main()
    else:
        print("⚠️ Please provide --prompts and --rubric to run the CLI evaluator.")


if __name__ == "__main__":
    main()
