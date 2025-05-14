# mental_health_llm_eval: Red-teaming and Evaluation Framework for Mental Health LLMs

This program provides a lightweight, modular evaluation framework for assessing the performance of large language models (LLMs) in mental health-related contexts.
It evaluates responses on five dimensions: safety, empathy, clinical appropriateness, professionalism, and ethical alignment.

It supports rule-based red flag detection (e.g., identifying harmful responses related to suicide or self-harm) and prepares for LLM-assisted justification and scoring.

The structure includes functions to load prompts, send them to models, check for dangerous content, assign structured scores, allow human-in-the-loop scoring via a simple Streamlit UI, and generate LLM-based justifications for model evaluations.


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

