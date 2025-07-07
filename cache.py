"""
Response caching system for LLM Service.
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from threading import Lock
import pickle

logger = logging.getLogger(__name__)


class ResponseCache:
    """Simple in-memory cache with optional disk persistence."""
    
    def __init__(self, 
                 enabled: bool = True,
                 max_size: int = 1000,
                 ttl_seconds: int = 3600,
                 persist_to_disk: bool = False,
                 cache_dir: Optional[str] = None):
        """
        Initialize cache.
        
        Args:
            enabled: Whether caching is enabled
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
            persist_to_disk: Whether to persist cache to disk
            cache_dir: Directory for disk cache (if persist_to_disk is True)
        """
        self.enabled = enabled
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.persist_to_disk = persist_to_disk
        
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_times: Dict[str, float] = {}
        self._hit_count = 0
        self._miss_count = 0
        self._lock = Lock()
        
        if persist_to_disk:
            self.cache_dir = Path(cache_dir or ".cache/llm_responses")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
            
    def get_key(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        """Generate cache key from prompt and parameters."""
        # Create a stable representation of the parameters
        param_str = json.dumps(params, sort_keys=True)
        
        # Combine all inputs
        combined = f"{model}:{prompt}:{param_str}"
        
        # Create hash
        return hashlib.sha256(combined.encode()).hexdigest()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if not self.enabled:
            return None
            
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                
                # Check if expired
                if time.time() - timestamp > self.ttl_seconds:
                    del self._cache[key]
                    del self._access_times[key]
                    self._miss_count += 1
                    return None
                    
                # Update access time
                self._access_times[key] = time.time()
                self._hit_count += 1
                logger.debug(f"Cache hit for key {key[:8]}...")
                return value
                
            self._miss_count += 1
            return None
            
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        if not self.enabled:
            return
            
        with self._lock:
            # Evict oldest items if at capacity
            if len(self._cache) >= self.max_size:
                self._evict_lru()
                
            # Store with timestamp
            self._cache[key] = (value, time.time())
            self._access_times[key] = time.time()
            
            # Persist to disk if enabled
            if self.persist_to_disk:
                self._save_entry_to_disk(key, value)
                
            logger.debug(f"Cached response for key {key[:8]}...")
            
    def _evict_lru(self) -> None:
        """Evict least recently used items."""
        # Find LRU item
        if not self._access_times:
            return
            
        lru_key = min(self._access_times, key=self._access_times.get)
        
        # Remove from cache
        del self._cache[lru_key]
        del self._access_times[lru_key]
        
        # Remove from disk if persisted
        if self.persist_to_disk:
            cache_file = self.cache_dir / f"{lru_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
                
        logger.debug(f"Evicted LRU entry {lru_key[:8]}...")
        
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._hit_count = 0
            self._miss_count = 0
            
            # Clear disk cache
            if self.persist_to_disk:
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                    
        logger.info("Cache cleared")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "enabled": self.enabled,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
            "persist_to_disk": self.persist_to_disk,
        }
        
    def _save_entry_to_disk(self, key: str, value: Any) -> None:
        """Save cache entry to disk."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "value": value,
                    "timestamp": time.time()
                }, f)
        except Exception as e:
            logger.error(f"Failed to save cache entry to disk: {e}")
            
    def _load_from_disk(self) -> None:
        """Load cache entries from disk."""
        if not self.cache_dir.exists():
            return
            
        loaded = 0
        expired = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    
                # Check if expired
                if time.time() - data["timestamp"] > self.ttl_seconds:
                    cache_file.unlink()
                    expired += 1
                    continue
                    
                # Add to cache
                key = cache_file.stem
                self._cache[key] = (data["value"], data["timestamp"])
                self._access_times[key] = data["timestamp"]
                loaded += 1
                
            except Exception as e:
                logger.error(f"Failed to load cache file {cache_file}: {e}")
                cache_file.unlink()
                
        if loaded > 0:
            logger.info(f"Loaded {loaded} cache entries from disk ({expired} expired)")


class SemanticCache(ResponseCache):
    """Cache that uses semantic similarity for matching."""
    
    def __init__(self, 
                 similarity_threshold: float = 0.95,
                 embedding_model: Optional[str] = None,
                 **kwargs):
        """
        Initialize semantic cache.
        
        Args:
            similarity_threshold: Minimum similarity score for cache hit
            embedding_model: Model to use for embeddings (optional)
            **kwargs: Arguments passed to ResponseCache
        """
        super().__init__(**kwargs)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self._embeddings: Dict[str, Any] = {}
        
        # Lazy load embedding model when needed
        self._embed_fn = None
        
    def _get_embedding(self, text: str) -> Any:
        """Get embedding for text."""
        if self._embed_fn is None:
            # Initialize embedding function
            # This is a placeholder - you'd use actual embedding model
            logger.warning("Semantic cache requires embedding model setup")
            return None
            
        return self._embed_fn(text)
        
    def get_semantic(self, prompt: str, model: str, params: Dict[str, Any], 
                    exact_match_params: bool = True) -> Optional[Any]:
        """Get item from cache using semantic similarity."""
        if not self.enabled:
            return None
            
        # First try exact match
        key = self.get_key(prompt, model, params)
        exact_result = self.get(key)
        if exact_result:
            return exact_result
            
        # If no exact match, try semantic similarity
        prompt_embedding = self._get_embedding(prompt)
        if prompt_embedding is None:
            return None
            
        with self._lock:
            best_match = None
            best_score = 0
            
            for cached_key, embedding in self._embeddings.items():
                # Check if parameters match (if required)
                if exact_match_params:
                    # This would require storing params with embeddings
                    pass
                    
                # Calculate similarity
                # Placeholder - actual implementation would use cosine similarity
                score = 0.0
                
                if score >= self.similarity_threshold and score > best_score:
                    best_score = score
                    best_match = cached_key
                    
            if best_match:
                return self.get(best_match)
                
        return None