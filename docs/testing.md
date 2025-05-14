---
layout: default
title: Testing Guide
---

# Testing Guide

This guide outlines how to test the Mental Health LLM Evaluation Framework comprehensively.

## Prerequisites

Before running tests, ensure you have:

1. Installed development dependencies:
   ```bash
   pip install "mental-health-llm-eval[dev]"
   ```

2. Set up test environment variables:
   ```bash
   export OPENAI_API_KEY=your_test_key
   export ANTHROPIC_API_KEY=your_test_key
   export HUGGINGFACE_TOKEN=your_test_token
   ```

## Test Categories

### 1. Unit Tests

Run the full test suite:
```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/test_evaluator.py    # Core evaluator tests
pytest tests/test_scorer.py       # Scoring system tests
pytest tests/test_red_flags.py    # Red flag detection tests
pytest tests/test_validation.py   # Input validation tests
```

### 2. Integration Tests

Test model backends:
```bash
# Test with mock backend
pytest tests/integration/test_mock_backend.py

# Test with real APIs (requires API keys)
pytest tests/integration/test_openai_backend.py
pytest tests/integration/test_anthropic_backend.py
pytest tests/integration/test_huggingface_backend.py
```

### 3. End-to-End Testing

1. **CLI Testing**:
   ```bash
   # Test single prompt evaluation
   mh-llm-eval evaluate "I'm feeling very anxious"
   
   # Test batch evaluation
   mh-llm-eval evaluate-batch prompts/prompt_bank.yaml
   
   # Test report generation
   mh-llm-eval report output/results.json
   ```

2. **UI Testing**:
   ```bash
   # Launch UI
   mh-llm-eval-ui
   
   # Access http://localhost:8501
   ```

### 4. Performance Testing

1. **Rate Limiting**:
   ```bash
   pytest tests/performance/test_rate_limits.py
   ```

2. **Load Testing**:
   ```bash
   pytest tests/performance/test_batch_processing.py
   ```

## Test Data

1. **Prompt Bank**:
   - Located in `prompts/prompt_bank.yaml`
   - Contains diverse test cases
   - Covers various mental health scenarios

2. **Expected Responses**:
   - Located in `tests/data/expected_responses.json`
   - Baseline for response evaluation
   - Includes edge cases

## Safety Testing

Test safety features:

1. **Input Validation**:
   - Malformed prompts
   - XSS attempts
   - SQL injection attempts

2. **Response Validation**:
   - Harmful content detection
   - Crisis response handling
   - Resource provision checking

3. **Rate Limiting**:
   - API quota management
   - Burst protection
   - Cooldown periods

## Continuous Integration

GitHub Actions workflow runs:
1. Unit tests
2. Integration tests (mock backend)
3. Linting and type checking
4. Documentation building

## Manual Testing Checklist

1. **Model Backend Testing**:
   - [ ] Test each backend independently
   - [ ] Verify error handling
   - [ ] Check token usage tracking
   - [ ] Test capability reporting

2. **UI Testing**:
   - [ ] Test all form inputs
   - [ ] Verify visualizations
   - [ ] Check responsiveness
   - [ ] Test error messages

3. **CLI Testing**:
   - [ ] Test all commands
   - [ ] Verify output formats
   - [ ] Check error handling
   - [ ] Test configuration options

4. **Documentation Testing**:
   - [ ] Verify all links work
   - [ ] Test code examples
   - [ ] Check API references
   - [ ] Validate installation steps

## Reporting Issues

1. Check existing issues on GitHub
2. Include:
   - Test environment details
   - Steps to reproduce
   - Expected vs actual results
   - Relevant logs

## Best Practices

1. Write tests for new features
2. Maintain test coverage above 80%
3. Use meaningful test names
4. Document test requirements
5. Include edge cases
6. Test error conditions 