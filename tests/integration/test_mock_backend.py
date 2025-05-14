"""Integration tests for the mock model backend."""

import pytest
from mental_health_llm_eval.model_backends import ModelBackendType, create_backend

@pytest.fixture
def mock_backend():
    """Create a configured mock backend."""
    config = {
        "delay": 0.1,
        "error_rate": 0.0
    }
    backend = create_backend(ModelBackendType.MOCK, config)
    return backend

def test_mock_backend_initialization(mock_backend):
    """Test mock backend initialization."""
    assert mock_backend.initialized
    assert mock_backend.delay == 0.1
    assert mock_backend.error_rate == 0.0

def test_mock_backend_query(mock_backend):
    """Test mock backend query functionality."""
    prompt = "I'm feeling anxious"
    system_prompt = "You are a helpful assistant"
    
    response = mock_backend.query(prompt, system_prompt=system_prompt)
    
    assert response.text
    assert not response.error
    assert "model" in response.metadata
    assert "usage" in response.metadata
    assert "mock_info" in response.metadata

def test_mock_backend_capabilities(mock_backend):
    """Test mock backend capabilities reporting."""
    capabilities = mock_backend.get_capabilities()
    
    assert "streaming" in capabilities
    assert "system_prompts" in capabilities
    assert "function_calling" in capabilities
    assert "is_mock" in capabilities
    assert capabilities["is_mock"] is True

def test_mock_backend_error_handling():
    """Test mock backend error handling."""
    # Test with invalid config
    with pytest.raises(ValueError):
        create_backend(ModelBackendType.MOCK, {"delay": "invalid"})
    
    # Test with high error rate
    error_backend = create_backend(ModelBackendType.MOCK, {"error_rate": 1.0})
    response = error_backend.query("test")
    assert response.error

def test_mock_backend_deterministic():
    """Test mock backend deterministic responses."""
    backend = create_backend(ModelBackendType.MOCK, {"delay": 0})
    prompt = "test prompt"
    
    response1 = backend.query(prompt)
    response2 = backend.query(prompt)
    
    assert response1.text == response2.text 