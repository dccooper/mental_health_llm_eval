# mental_health_llm_eval: Red-teaming and Evaluation Framework for Mental Health LLMs
"""
This program provides a lightweight, modular evaluation framework for assessing the performance of large language models (LLMs) in mental health-related contexts.
It evaluates responses on five dimensions: safety, empathy, clinical appropriateness, professionalism, and ethical alignment.

It supports rule-based red flag detection (e.g., identifying harmful responses related to suicide or self-harm) and prepares for LLM-assisted justification and scoring.

The structure includes functions to load prompts, send them to models, check for dangerous content, assign structured scores, allow human-in-the-loop scoring via a simple Streamlit UI, and generate LLM-based justifications for model evaluations.
"""

## 🚀 How to Use

This toolkit supports multiple modes of operation. You can run it via the command line using:

```bash
python -m src [mode options] --prompts path/to/prompts.yaml --rubric path/to/rubric.yaml
```

### 🔧 Options

| Flag         | Description                                                             |
|--------------|-------------------------------------------------------------------------|
| `--cli`      | Runs batch evaluation over all prompts using CLI (default behavior)     |
| `--ui`       | Launches the Streamlit human-in-the-loop scoring interface              |
| `--validate` | Validates prompt and rubric YAML files                                  |
| `--version`  | Prints the tool version and exits                                       |
| `--prompts`  | Path to your YAML prompt bank (used by CLI or validate modes)           |
| `--rubric`   | Path to your YAML scoring rubric (used by CLI or validate modes)        |

### 🧪 Examples

**Run full CLI evaluation:**
```bash
python -m src --cli --prompts prompts/prompt_bank.yaml --rubric rubric/scoring_rubric.yaml
```

**Launch the interactive Streamlit UI:**
```bash
python -m src --ui
```

**Check if your YAML files are properly formatted:**
```bash
python -m src --validate --prompts prompts/prompt_bank.yaml --rubric rubric/scoring_rubric.yaml
```

**Print version:**
```bash
python -m src --version
```


# ---------- cli/main.py ----------

"""
Command-line interface to run batch evaluations of LLM responses for mental health safety and quality.
"""

import argparse
import json
import pandas as pd
from src.evaluator import run_evaluation


def save_results(results, output_base: str):
    """
    Saves evaluation results to CSV and JSON files.

    Args:
        results (list): The list of evaluation dictionaries.
        output_base (str): Base filename (without extension).
    """
    # Save JSON
    with open(f"{output_base}.json", "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)

    # Save CSV
    df = pd.json_normalize(results)
    df.to_csv(f"{output_base}.csv", index=False)


def main():
    """
    Command-line entry point for batch evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate mental health-related LLM responses.")
    parser.add_argument("--prompts", required=True, help="Path to the YAML prompt bank")
    parser.add_argument("--rubric", required=True, help="Path to the YAML scoring rubric")
    parser.add_argument("--output", default="eval_results", help="Base filename for output files")

    args = parser.parse_args()
    results = run_evaluation(args.prompts, args.rubric)
    save_results(results, args.output)
    print(f"Saved evaluation results to {args.output}.json and {args.output}.csv")


if __name__ == "__main__":
    main()
