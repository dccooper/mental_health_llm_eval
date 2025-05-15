#!/usr/bin/env python3
# __main__.py — Unified entry point for all evaluation modes
#
# Supports multiple execution paths:
#   --cli       → batch evaluator
#   --ui        → launch Streamlit app
#   --validate  → check YAML formats
#   --version   → print version number

import sys
import argparse
import subprocess
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            from .cli import main as cli_main
from .evaluator import load_prompts
from .scorer import load_rubric

TOOL_VERSION = "0.1.0"

def launch_ui():
    """Starts the Streamlit app."""
    print("Launching UI...")
    subprocess.run(["streamlit", "run", "app/streamlit_app.py"])

def validate_inputs(prompt_path, rubric_path):
    """Checks if prompt and rubric YAML files load correctly."""
    try:
        load_prompts(prompt_path)
        print(f"✅ Prompts OK: {prompt_path}")
    except Exception as e:
        print(f"❌ Prompts error: {e}")

    try:
        load_rubric(rubric_path)
        print(f"✅ Rubric OK: {rubric_path}")
    except Exception as e:
        print(f"❌ Rubric error: {e}")

def main():
    """Main command-line router."""
    parser = argparse.ArgumentParser(description="Mental Health LLM Eval Toolkit")
    parser.add_argument("--cli", action="store_true", help="Run CLI batch evaluator")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit UI")
    parser.add_argument("--validate", action="store_true", help="Validate prompt/rubric YAMLs")
    parser.add_argument("--version", action="store_true", help="Print version")
    parser.add_argument("--prompts", help="Path to prompt_bank.yaml")
    parser.add_argument("--rubric", help="Path to scoring_rubric.yaml")

    args = parser.parse_args()

    if args.version:
        print(f"Mental Health Eval v{TOOL_VERSION}")
        return

    if args.ui:
        launch_ui()
        return

    if args.validate:
        if not args.prompts or not args.rubric:
            print("⚠️ Provide both --prompts and --rubric for validation.")
        else:
            validate_inputs(args.prompts, args.rubric)
        return

    if args.cli:
        if not args.prompts or not args.rubric:
            print("⚠️ Provide --prompts and --rubric for CLI evaluation.")
        else:
            cli_main.main()
        return

    print("📌 Use --cli, --ui, --validate, or --version to run this tool.")

if __name__ == "__main__":
    main()
