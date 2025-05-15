---
layout: default
title: Home
---

# Mental Health LLM Evaluation Framework

A lightweight, modular framework for evaluating LLM responses in mental health contexts.

## Overview

This framework provides tools and methodologies for assessing the performance of large language models (LLMs) in mental health-related contexts. It evaluates responses on five key dimensions:

- Safety
- Empathy
- Clinical appropriateness
- Professionalism
- Ethical alignment

## Key Features

### Multiple Model Support
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Local models (via HuggingFace)
- Custom model integration

### Comprehensive Evaluation
- Safety checks and red flag detection
- Clinical appropriateness scoring
- Empathy and tone analysis
- Response validation
- Detailed scoring rubrics
- Customizable evaluation criteria
- CSV-based scoring rubric templates

### Safety First
- Input validation and sanitization
- Rate limiting
- Content warnings
- Audit logging
- Real-time red flag detection
- Severity-based warning system

### User-Friendly Interface
- Web-based UI for evaluations
- Command-line interface
- Batch processing support
- Result visualization
- Interactive scoring interface
- Session statistics tracking
- Export functionality (CSV/JSON)
- Step-by-step guidance

## Documentation

### For Non-Technical Users
- [Simple Step-by-Step Guide](simple_instructions.md) - A beginner-friendly guide for using the tool
- [Installation for Beginners](installation.md#for-beginners) - Easy-to-follow installation instructions

### Technical Documentation
- [Installation Guide](installation.md) - Detailed setup instructions
- [Testing Guide](testing.md) - Information about running and writing tests
- [Prompt Format](prompt_format.md) - Guide to creating and formatting prompts
- [Evaluator Configuration](evaluator_prompt.md) - Details about configuring the evaluator

### Development
- [Contributing Guidelines](contributing.md)
- [API Reference](api.md)
- [GitHub Repository](https://github.com/dccooper/mental_health_llm_eval)

## Recent Updates

- Added customizable scoring rubrics via CSV templates
- Improved UI with clear step-by-step instructions
- Enhanced sidebar organization for better workflow
- Added session statistics tracking
- Improved error handling and user feedback
- Added beginner-friendly documentation

## Disclaimer

This tool is designed to assist in evaluating LLM responses in mental health contexts. It is not a substitute for professional clinical judgment. Always defer to qualified mental health professionals for actual patient care decisions. 