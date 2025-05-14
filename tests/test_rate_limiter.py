"""
Tests for the rate limiting module.
"""

import pytest
import time
from src.rate_limiter import (
    TokenBucket,
    SlidingWindowLimiter,
    RateLimiter,
    LimitType,
    RateLimitExceeded
)

def test_token_bucket():
    """Test token bucket rate limiter."""
    # Create a bucket with 10 tokens per second, max 10 tokens
    bucket = TokenBucket(rate=10, capacity=10)
    
    # Should be able to get all tokens initially
    assert bucket.try_acquire(10)
    
    # Should not be able to get any more tokens immediately
    assert not bucket.try_acquire(1)
    
    # Wait for 0.5 seconds, should get 5 tokens back
    time.sleep(0.5)
    assert bucket.try_acquire(5)
    assert not bucket.try_acquire(1)
    
    # Wait for bucket to refill completely
    time.sleep(1.0)
    assert bucket.try_acquire(10)

def test_sliding_window():
    """Test sliding window rate limiter."""
    # Allow 5 requests per second
    limiter = SlidingWindowLimiter(window_size=1, max_requests=5)
    
    # Should allow 5 requests
    for _ in range(5):
        assert limiter.try_acquire()
    
    # 6th request should fail
    assert not limiter.try_acquire()
    
    # Wait for window to slide
    time.sleep(1.1)
    assert limiter.try_acquire()

def test_rate_limiter():
    """Test the main rate limiter class."""
    limiter = RateLimiter()
    
    # Test prompt rate limiting
    for _ in range(10):  # Burst size for prompts
        limiter.try_acquire(LimitType.PROMPT)
    
    # Next prompt should fail (burst limit exceeded)
    with pytest.raises(RateLimitExceeded) as exc:
        limiter.try_acquire(LimitType.PROMPT)
    assert "Burst limit exceeded" in str(exc.value)
    
    # Test model call rate limiting
    for _ in range(5):  # Burst size for model calls
        limiter.try_acquire(LimitType.MODEL_CALL)
    
    # Next model call should fail
    with pytest.raises(RateLimitExceeded) as exc:
        limiter.try_acquire(LimitType.MODEL_CALL)
    assert "Burst limit exceeded" in str(exc.value)

def test_rate_limiter_status():
    """Test rate limiter status reporting."""
    limiter = RateLimiter()
    
    # Check initial status
    status = limiter.get_status(LimitType.PROMPT)
    assert status['total_limit'] == 100
    assert status['burst_size'] == 10
    assert status['current_usage'] == 0
    assert status['available_tokens'] == 10
    
    # Use some capacity
    for _ in range(5):
        limiter.try_acquire(LimitType.PROMPT)
    
    # Check updated status
    status = limiter.get_status(LimitType.PROMPT)
    assert status['current_usage'] == 5
    assert status['available_tokens'] == 5

def test_concurrent_access():
    """Test thread safety of rate limiters."""
    import threading
    
    limiter = RateLimiter()
    success_count = {'value': 0}
    thread_count = 20
    
    def try_acquire():
        try:
            limiter.try_acquire(LimitType.VALIDATION)
            success_count['value'] += 1
        except RateLimitExceeded:
            pass
    
    # Launch threads simultaneously
    threads = []
    for _ in range(thread_count):
        thread = threading.Thread(target=try_acquire)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Should only succeed burst_size times
    assert success_count['value'] == 20  # Validation burst size

def test_different_limit_types():
    """Test that different limit types are independent."""
    limiter = RateLimiter()
    
    # Use up prompt burst capacity
    for _ in range(10):
        limiter.try_acquire(LimitType.PROMPT)
    
    # Should still be able to do validation
    limiter.try_acquire(LimitType.VALIDATION)
    
    # Should still be able to do model calls
    limiter.try_acquire(LimitType.MODEL_CALL)

def test_rate_limit_recovery():
    """Test that rate limits recover over time."""
    limiter = RateLimiter()
    
    # Use up burst capacity
    for _ in range(5):
        limiter.try_acquire(LimitType.MODEL_CALL)
    
    # Wait for some recovery
    time.sleep(1.0)
    
    # Should be able to make a request again
    limiter.try_acquire(LimitType.MODEL_CALL)

@pytest.mark.parametrize("limit_type", [
    LimitType.PROMPT,
    LimitType.MODEL_CALL,
    LimitType.VALIDATION
])
def test_all_limit_types(limit_type):
    """Test that all limit types work correctly."""
    limiter = RateLimiter()
    
    # Get the burst size for this type
    burst_size = limiter.limits[limit_type].burst_size
    
    # Should be able to use burst capacity
    for _ in range(burst_size):
        limiter.try_acquire(limit_type)
    
    # Next request should fail
    with pytest.raises(RateLimitExceeded):
        limiter.try_acquire(limit_type) 