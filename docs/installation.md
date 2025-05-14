---
layout: default
title: Installation
---

# Installation Guide

## Prerequisites

Before installing the Mental Health LLM Evaluation Framework, ensure you have:

- Python 3.8 or higher
- pip (Python package installer)
- Optional: CUDA-capable GPU for local models

## Installation Methods

### 1. Basic Installation

```bash
pip install mental-health-llm-eval
```

### 2. Installation with Features

#### With UI Support
```bash
pip install "mental-health-llm-eval[ui]"
```

#### With Specific Model Backend
```bash
# OpenAI backend
pip install "mental-health-llm-eval[openai]"

# Anthropic backend
pip install "mental-health-llm-eval[anthropic]"

# HuggingFace backend
pip install "mental-health-llm-eval[huggingface]"
```

#### Full Installation
```bash
pip install "mental-health-llm-eval[ui,openai,anthropic,huggingface]"
```

## Configuration

1. Create a `.env` file in your project directory:

```env
# Required for OpenAI backend
OPENAI_API_KEY=your_api_key_here

# Required for Anthropic backend
ANTHROPIC_API_KEY=your_api_key_here

# Optional: HuggingFace token for private models
HUGGINGFACE_TOKEN=your_token_here
```

2. Optional: Configure GPU Support

If you plan to use local models with GPU acceleration:

- Install CUDA Toolkit (for NVIDIA GPUs)
- Install appropriate PyTorch version with CUDA support
- Set environment variable: `CUDA_VISIBLE_DEVICES`

## Development Installation

For contributors and developers:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mental_health_llm_eval.git
cd mental_health_llm_eval
```

2. Install development dependencies:
```bash
make setup-dev
```

## Verification

To verify your installation:

1. Run the CLI tool:
```bash
mh-llm-eval --version
```

2. Launch the UI:
```bash
mh-llm-eval-ui
```

## Troubleshooting

Common issues and solutions:

1. **GPU Not Detected**
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"` 

2. **API Key Issues**
   - Ensure `.env` file is in the correct directory
   - Check environment variables are loaded
   - Verify API key validity

3. **Package Conflicts**
   - Create a new virtual environment
   - Update pip: `pip install --upgrade pip`
   - Install with specific versions if needed

For more help, visit our [GitHub Issues](https://github.com/yourusername/mental_health_llm_eval/issues) page. 