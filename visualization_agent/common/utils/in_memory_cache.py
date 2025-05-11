"""In Memory Cache utility with LRU eviction."""

import threading
import time
from typing import Any, Dict, Optional, List
from collections import OrderedDict
import functools


class InMemoryCache:
    """A thread-safe Singleton class to manage cache data with LRU eviction.

    Ensures only one instance of the cache exists across the application.
    Implements LRU (Least Recently Used) eviction when cache exceeds maxsize.
    """

    _instance: Optional["InMemoryCache"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls):
        """Override __new__ to control instance creation (Singleton pattern).

        Uses a lock to ensure thread safety during the first instantiation.

        Returns:
            The singleton instance of InMemoryCache.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, maxsize: int = 100):
        """Initialize the cache storage with LRU tracking.

        Uses a flag (_initialized) to ensure this logic runs only on the very first
        creation of the singleton instance.

        Args:
            maxsize: Maximum number of items to keep in the cache before evicting.
        """
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    # Store cache data in an OrderedDict for LRU tracking
                    self._cache_data: OrderedDict[str, Any] = OrderedDict()
                    self._ttl: Dict[str, float] = {}
                    self._data_lock: threading.Lock = threading.Lock()
                    self._maxsize: int = maxsize
                    self._initialized = True

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a key-value pair, evicting oldest items if cache is full.

        Args:
            key: The key for the data.
            value: The data to store.
            ttl: Time to live in seconds. If None, data will not expire.
        """
        with self._data_lock:
            # Remove existing key from the ordered dict if it exists
            if key in self._cache_data:
                del self._cache_data[key]
            
            # Add new item (will be at the end of the OrderedDict)
            self._cache_data[key] = value

            # Set TTL if provided
            if ttl is not None:
                self._ttl[key] = time.time() + ttl
            else:
                if key in self._ttl:
                    del self._ttl[key]
                    
            # Perform LRU eviction if necessary
            self._enforce_size_limit()
    
    def _enforce_size_limit(self) -> None:
        """Remove oldest entries when cache exceeds size limit."""
        while len(self._cache_data) > self._maxsize:
            # Get the oldest key (first item in OrderedDict)
            oldest_key, _ = next(iter(self._cache_data.items()))
            
            # Remove the oldest item
            del self._cache_data[oldest_key]
            if oldest_key in self._ttl:
                del self._ttl[oldest_key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value associated with a key.
        
        If the key exists, it will be moved to the end of the OrderedDict,
        making it the most recently used item.

        Args:
            key: The key for the data within the session.
            default: The value to return if the session or key is not found.

        Returns:
            The cached value, or the default value if not found.
        """
        with self._data_lock:
            # Check TTL first
            if key in self._ttl and time.time() > self._ttl[key]:
                del self._cache_data[key]
                del self._ttl[key]
                return default
                
            # If key exists, move it to the end (mark as most recently used)
            if key in self._cache_data:
                value = self._cache_data[key]
                del self._cache_data[key]  # Remove and re-add to move to end
                self._cache_data[key] = value
                return value
                
            return default

    def delete(self, key: str) -> bool:
        """Delete a specific key-value pair from a cache.

        Args:
            key: The key to delete.

        Returns:
            True if the key was found and deleted, False otherwise.
        """
        with self._data_lock:
            if key in self._cache_data:
                del self._cache_data[key]
                if key in self._ttl:
                    del self._ttl[key]
                return True
            return False

    def clear(self) -> bool:
        """Remove all data.

        Returns:
            True if the data was cleared, False otherwise.
        """
        with self._data_lock:
            self._cache_data.clear()
            self._ttl.clear()
            return True
        return False
    
    def set_maxsize(self, maxsize: int) -> None:
        """Update the maximum size of the cache.
        
        Args:
            maxsize: New maximum size for the cache.
        """
        with self._data_lock:
            self._maxsize = maxsize
            self._enforce_size_limit()
    
    def get_size(self) -> int:
        """Get the current size of the cache.
        
        Returns:
            Number of items in the cache.
        """
        with self._data_lock:
            return len(self._cache_data)
    
    def get_maxsize(self) -> int:
        """Get the maximum size of the cache.
        
        Returns:
            Maximum number of items allowed in the cache.
        """
        return self._maxsize
