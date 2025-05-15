"""
Rate Limiting Module
==================

This module provides rate limiting functionality to protect the system from overuse.
It implements both token bucket and sliding window algorithms for different rate limiting needs:

1. Token Bucket: For controlling burst rates while allowing some flexibility
2. Sliding Window: For strict rate limiting over time periods

The module is thread-safe and can be used across different components of the system.
"""

import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum


class RateLimitExceeded(Exception):
    """Raised when a rate limit is exceeded."""
    pass


class LimitType(Enum):
    """Types of rate limits that can be applied."""
    PROMPT = "prompt"          # Limit on prompt evaluations
    MODEL_CALL = "model"       # Limit on model API calls
    VALIDATION = "validation"  # Limit on validation operations


@dataclass
class RateLimit:
    """Configuration for a rate limit."""
    requests: int             # Number of requests allowed
    time_window: int         # Time window in seconds
    burst_size: Optional[int] = None  # Maximum burst size (for token bucket)


class TokenBucket:
    """
    Token bucket rate limiter implementation.
    Allows for bursts of requests while maintaining a long-term rate limit.
    """
    
    def __init__(self, rate: float, capacity: int):
        """
        Initialize a token bucket.
        
        Args:
            rate: Token refill rate per second
            capacity: Maximum number of tokens the bucket can hold
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_update = now
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class SlidingWindowLimiter:
    """
    Sliding window rate limiter implementation.
    Provides strict rate limiting over a time window.
    """
    
    def __init__(self, window_size: int, max_requests: int):
        """
        Initialize a sliding window rate limiter.
        
        Args:
            window_size: Size of the sliding window in seconds
            max_requests: Maximum number of requests allowed in the window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests: List[float] = []
        self._lock = threading.Lock()
    
    def _cleanup_old_requests(self) -> None:
        """Remove requests outside the current window."""
        now = time.time()
        cutoff = now - self.window_size
        self.requests = [t for t in self.requests if t > cutoff]
    
    def try_acquire(self) -> bool:
        """
        Try to record a new request.
        
        Returns:
            True if request is allowed, False otherwise
        """
        with self._lock:
            self._cleanup_old_requests()
            if len(self.requests) < self.max_requests:
                self.requests.append(time.time())
                return True
            return False


class RateLimiter:
    """
    Main rate limiter class that combines token bucket and sliding window approaches.
    Provides rate limiting for different types of operations with different strategies.
    """
    
    def __init__(self):
        """Initialize rate limiters for different operation types."""
        # Default rate limits
        self.limits = {
            LimitType.PROMPT: RateLimit(
                requests=100,    # 100 prompts
                time_window=3600,  # per hour
                burst_size=10     # max 10 at once
            ),
            LimitType.MODEL_CALL: RateLimit(
                requests=600,    # 600 calls
                time_window=3600,  # per hour
                burst_size=5      # max 5 at once
            ),
            LimitType.VALIDATION: RateLimit(
                requests=1000,   # 1000 validations
                time_window=3600,  # per hour
                burst_size=20     # max 20 at once
            )
        }
        
        # Initialize limiters for each type
        self.token_buckets: Dict[LimitType, TokenBucket] = {}
        self.sliding_windows: Dict[LimitType, SlidingWindowLimiter] = {}
        
        for limit_type, config in self.limits.items():
            # Token bucket for burst control
            self.token_buckets[limit_type] = TokenBucket(
                rate=config.requests / config.time_window,
                capacity=config.burst_size or config.requests
            )
            
            # Sliding window for overall rate limiting
            self.sliding_windows[limit_type] = SlidingWindowLimiter(
                window_size=config.time_window,
                max_requests=config.requests
            )
    
    def check_rate_limit(self, limit_type: LimitType) -> Tuple[bool, Optional[str]]:
        """
        Check if an operation would exceed rate limits.
        
        Args:
            limit_type: Type of operation to check
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        # Check burst limit
        if not self.token_buckets[limit_type].try_acquire():
            return False, f"Burst limit exceeded for {limit_type.value} operations"
        
        # Check overall rate limit
        if not self.sliding_windows[limit_type].try_acquire():
            return False, f"Rate limit exceeded for {limit_type.value} operations"
        
        return True, None
    
    def try_acquire(self, limit_type: LimitType) -> None:
        """
        Try to acquire permission for an operation, raising an exception if not allowed.
        
        Args:
            limit_type: Type of operation to perform
            
        Raises:
            RateLimitExceeded: If the operation would exceed rate limits
        """
        is_allowed, error_msg = self.check_rate_limit(limit_type)
        if not is_allowed:
            raise RateLimitExceeded(error_msg)
    
    def get_status(self, limit_type: LimitType) -> Dict[str, Any]:
        """
        Get current rate limit status for an operation type.
        
        Args:
            limit_type: Type of operation to check
            
        Returns:
            Dictionary with current rate limit status
        """
        window = self.sliding_windows[limit_type]
        bucket = self.token_buckets[limit_type]
        limit = self.limits[limit_type]
        
        return {
            'total_limit': limit.requests,
            'time_window': limit.time_window,
            'burst_size': limit.burst_size,
            'current_usage': len(window.requests),
            'available_tokens': bucket.tokens,
            'reset_time': time.time() + limit.time_window
        } 