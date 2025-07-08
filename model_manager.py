"""
Model Manager for handling different inference backends (llama.cpp and MLX).
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
from threading import Lock

from embedders import create_embedder, BaseEmbedder, EmbeddingResult
from rerankers import create_reranker, BaseReranker

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model tiers for different computational requirements."""
    LIGHT = "light"
    MEDIUM = "medium" 
    HEAVY = "heavy"
    EMBEDDING = "embedding"  # Dedicated tier for embedding models
    RERANKER = "reranker"    # Dedicated tier for reranking models


class InferenceBackend(Enum):
    """Supported inference backends."""
    LLAMACPP = "llamacpp"
    MLX = "mlx"


@dataclass
class ModelConfig:
    """Configuration for a model."""
    path: str
    tier: ModelTier
    backend: InferenceBackend
    
    # Model parameters with defaults
    n_ctx: int = 2048
    n_batch: int = 512
    n_threads: Optional[int] = None
    n_gpu_layers: int = -1  # -1 for all layers on GPU
    
    # Generation defaults (can be overridden at runtime)
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 1000
    repeat_penalty: float = 1.1
    
    # Embedding-specific settings
    embedding_mode: bool = False  # Whether this model is primarily for embeddings
    embedding_dimension: Optional[int] = None  # Override embedding dimension
    
    # Backend-specific parameters
    use_mmap: bool = True
    use_mlock: bool = False
    verbose: bool = False
    
    @staticmethod
    def _parse_env(key: str, default: str = "") -> str:
        """Parse environment variable, stripping inline comments."""
        value = os.getenv(key, default)
        if '#' in value:
            # Strip inline comments
            value = value.split('#')[0].strip()
        return value
    
    @classmethod
    def from_env(cls, tier: ModelTier) -> Optional["ModelConfig"]:
        """Create model config from environment variables."""
        tier_upper = tier.value.upper()
        
        # Get model path
        model_path = cls._parse_env(f"{tier_upper}_MODEL_PATH")
        if not model_path:
            return None
            
        # Determine backend from file extension or env var
        backend_str = cls._parse_env(f"{tier_upper}_BACKEND", "").lower()
        if backend_str == "mlx" or model_path.endswith(".mlx"):
            backend = InferenceBackend.MLX
        else:
            backend = InferenceBackend.LLAMACPP
            
        # Get parameters with defaults
        return cls(
            path=model_path,
            tier=tier,
            backend=backend,
            n_ctx=int(cls._parse_env(f"{tier_upper}_N_CTX", "8192")),
            n_batch=int(cls._parse_env(f"{tier_upper}_N_BATCH", "512")),
            n_threads=int(cls._parse_env(f"{tier_upper}_N_THREADS", str(os.cpu_count() or 8))),
            n_gpu_layers=int(cls._parse_env(f"{tier_upper}_N_GPU_LAYERS", "-1")),
            temperature=float(cls._parse_env(f"{tier_upper}_TEMPERATURE", "0.7")),
            top_p=float(cls._parse_env(f"{tier_upper}_TOP_P", "0.95")),
            top_k=int(cls._parse_env(f"{tier_upper}_TOP_K", "40")),
            max_tokens=int(cls._parse_env(f"{tier_upper}_MAX_TOKENS", "1000")),
            repeat_penalty=float(cls._parse_env(f"{tier_upper}_REPEAT_PENALTY", "1.1")),
            use_mmap=cls._parse_env(f"{tier_upper}_USE_MMAP", "true").lower() == "true",
            use_mlock=cls._parse_env(f"{tier_upper}_USE_MLOCK", "false").lower() == "true",
            embedding_mode=cls._parse_env(f"{tier_upper}_EMBEDDING_MODE", "false").lower() == "true",
            embedding_dimension=int(cls._parse_env(f"{tier_upper}_EMBEDDING_DIMENSION", "0")) or None,
        )


class BaseModelWrapper:
    """Base class for model wrappers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.embedder: Optional[BaseEmbedder] = None
        self.reranker: Optional[BaseReranker] = None
        
    async def initialize(self):
        """Initialize the model."""
        raise NotImplementedError
        
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text from prompt."""
        raise NotImplementedError
        
    def cleanup(self):
        """Cleanup resources."""
        pass


class LlamaCppWrapper(BaseModelWrapper):
    """Wrapper for llama.cpp models."""
    
    async def initialize(self):
        """Initialize llama.cpp model."""
        try:
            from llama_cpp import Llama
            
            logger.info(f"Loading llama.cpp model from {self.config.path}")
            start_time = time.time()
            
            # For embedding models, we need to set n_ubatch to handle input tokens
            init_params = {
                "model_path": self.config.path,
                "n_ctx": self.config.n_ctx,
                "n_batch": self.config.n_batch,
                "n_threads": self.config.n_threads,
                "n_gpu_layers": self.config.n_gpu_layers,
                "use_mmap": self.config.use_mmap,
                "use_mlock": self.config.use_mlock,
                "verbose": False,  # Use a default value
                "embedding": self.config.embedding_mode,  # Enable embedding mode if configured
            }
            
            # Set n_ubatch for embedding models to match n_ctx
            if self.config.embedding_mode:
                init_params["n_ubatch"] = self.config.n_ctx
            
            self.model = Llama(**init_params)
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {self.config.tier.value} model in {load_time:.2f}s")
            
            # Warmup - skip for embedding/reranker models as they don't generate text
            if self.config.tier not in [ModelTier.EMBEDDING, ModelTier.RERANKER]:
                self.model("Hello", max_tokens=1)
            
            # Create embedder/reranker if this is an embedding or reranker model
            if self.config.tier == ModelTier.EMBEDDING:
                self.embedder = create_embedder(self.model, self.config)
            elif self.config.tier == ModelTier.RERANKER:
                self.reranker = create_reranker(self.model, self.config)
            
        except ImportError:
            raise RuntimeError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        except Exception as e:
            logger.error(f"Failed to load llama.cpp model: {e}")
            raise
            
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate using llama.cpp."""
        if not self.model:
            raise RuntimeError("Model not initialized")
            
        # Extract parameters
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)
        top_k = kwargs.get("top_k", self.config.top_k)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        repeat_penalty = kwargs.get("repeat_penalty", self.config.repeat_penalty)
        stop = kwargs.get("stop", [])
        seed = kwargs.get("seed", -1)
        
        # Handle grammar
        grammar = None
        grammar_str = kwargs.get("grammar")
        if grammar_str:
            try:
                from llama_cpp import LlamaGrammar
                grammar = LlamaGrammar.from_string(grammar_str)
            except Exception as e:
                logger.warning(f"Failed to parse grammar: {e}")
                
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _generate():
            start_time = time.time()
            result = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop,
                seed=seed,
                grammar=grammar,
                echo=False,
            )
            
            generation_time = time.time() - start_time
            
            # Extract token counts
            completion_tokens = len(result["choices"][0]["text"].split())
            prompt_tokens = len(prompt.split())
            
            return {
                "text": result["choices"][0]["text"],
                "finish_reason": result["choices"][0].get("finish_reason", "stop"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "generation_time": generation_time,
            }
            
        return await loop.run_in_executor(None, _generate)


class MLXWrapper(BaseModelWrapper):
    """Wrapper for MLX models."""
    
    async def initialize(self):
        """Initialize MLX model."""
        try:
            import mlx
            import mlx.core as mx
            from mlx_lm import load, generate
            
            logger.info(f"Loading MLX model from {self.config.path}")
            start_time = time.time()
            
            # Load model and tokenizer
            self.model, self.tokenizer = load(self.config.path)
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {self.config.tier.value} MLX model in {load_time:.2f}s")
            
            # Store MLX functions
            self._generate = generate
            self._mx = mx
            
            # Create embedder/reranker if this is an embedding or reranker model
            if self.config.tier == ModelTier.EMBEDDING:
                self.embedder = create_embedder(self.model, self.config)
            elif self.config.tier == ModelTier.RERANKER:
                self.reranker = create_reranker(self.model, self.config)
            
        except ImportError:
            raise RuntimeError("MLX not installed. Install with: pip install mlx mlx-lm")
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            raise
            
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate using MLX."""
        if not self.model:
            raise RuntimeError("Model not initialized")
            
        # Extract parameters
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        repetition_penalty = kwargs.get("repeat_penalty", self.config.repeat_penalty)
        
        # MLX doesn't support all parameters that llama.cpp does
        # We'll use what's available
        
        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        
        def _generate():
            start_time = time.time()
            
            # Generate text
            try:
                # Try with 'temperature' parameter (newer versions)
                response = self._generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty,
                    verbose=False,
                )
            except TypeError as e:
                if "'temp'" in str(e):
                    # Fallback to 'temp' parameter (older versions)
                    response = self._generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        temp=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        repetition_penalty=repetition_penalty,
                        verbose=False,
                    )
                else:
                    raise
            
            generation_time = time.time() - start_time
            
            # Estimate token counts
            completion_tokens = len(response.split())
            prompt_tokens = len(prompt.split())
            
            return {
                "text": response,
                "finish_reason": "stop",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "generation_time": generation_time,
            }
            
        return await loop.run_in_executor(None, _generate)
    
    async def generate_embedding(self, text: str) -> Dict[str, Any]:
        """Generate embedding using MLX model."""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        loop = asyncio.get_event_loop()
        
        def _embed():
            import mlx.core as mx
            
            # Tokenize the input
            tokens = self.tokenizer.encode(text)
            tokens_mx = mx.array(tokens).reshape(1, -1)
            
            # Get model embeddings - this gets the last hidden states
            # For transformer models, we typically use the output embeddings
            with mx.no_grad():
                # Pass through the model to get hidden states
                outputs = self.model(tokens_mx)
                
                # Get embeddings - different strategies:
                # 1. Mean pooling over sequence length
                # 2. Use [CLS] token (first token)
                # 3. Use last token
                
                # Using mean pooling as default
                embeddings = outputs[0]  # Shape: [1, seq_len, hidden_dim]
                embedding = mx.mean(embeddings, axis=1).squeeze(0)  # Mean over sequence
                
                # Convert to numpy/list
                embedding_list = embedding.tolist()
            
            return {
                "embedding": embedding_list,
                "dimension": len(embedding_list)
            }
        
        return await loop.run_in_executor(None, _embed)


class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self):
        self.models: Dict[ModelTier, BaseModelWrapper] = {}
        self.configs: Dict[ModelTier, ModelConfig] = {}
        self._lock = Lock()
        self._loading: Dict[ModelTier, asyncio.Lock] = {}  # Async locks for model loading
        
    async def initialize(self):
        """Initialize configurations without loading models (lazy loading)."""
        configured_tiers = 0
        
        for tier in ModelTier:
            config = ModelConfig.from_env(tier)
            if config and Path(config.path).exists():
                self.configs[tier] = config
                self._loading[tier] = asyncio.Lock()  # Create async lock for each tier
                configured_tiers += 1
                logger.info(f"Configuration found for {tier.value} tier: {config.path}")
            else:
                logger.warning(f"No valid configuration found for {tier.value} tier")
                
        logger.info(f"Lazy loading enabled for {configured_tiers} model configurations")
        
    async def _load_model(self, tier: ModelTier, config: ModelConfig):
        """Load a single model."""
        try:
            # Create appropriate wrapper
            if config.backend == InferenceBackend.LLAMACPP:
                wrapper = LlamaCppWrapper(config)
            elif config.backend == InferenceBackend.MLX:
                wrapper = MLXWrapper(config)
            else:
                raise ValueError(f"Unknown backend: {config.backend}")
                
            # Initialize model
            await wrapper.initialize()
            
            with self._lock:
                self.models[tier] = wrapper
                
        except Exception as e:
            logger.error(f"Failed to load {tier.value} model: {e}")
            raise
    
    async def _ensure_model_loaded(self, tier: ModelTier) -> bool:
        """Ensure a model is loaded, loading it if necessary. Returns True if successful."""
        # Quick check without lock
        if tier in self.models:
            return True
            
        # Check if configuration exists
        if tier not in self.configs:
            return False
            
        # Use async lock to prevent concurrent loading of the same model
        async with self._loading[tier]:
            # Double-check after acquiring lock
            if tier in self.models:
                return True
                
            logger.info(f"Lazy loading {tier.value} model on first use...")
            try:
                await self._load_model(tier, self.configs[tier])
                return True
            except Exception as e:
                logger.error(f"Failed to lazy load {tier.value} model: {e}")
                return False
            
    async def generate(self, model_tier: ModelTier, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using specified model tier."""
        # Try to load the requested model
        if not await self._ensure_model_loaded(model_tier):
            # Try to find an alternative tier
            for tier in [ModelTier.MEDIUM, ModelTier.LIGHT, ModelTier.HEAVY]:
                if await self._ensure_model_loaded(tier):
                    logger.warning(f"Model tier {model_tier.value} not available, using {tier.value}")
                    model_tier = tier
                    break
            else:
                raise ValueError(f"No models available. Requested tier: {model_tier.value}")
                
        model = self.models[model_tier]
        return await model.generate(prompt, **kwargs)
        
    async def generate_embedding(self, text: str, tier: ModelTier, 
                                embedding_type: str = "dense", 
                                return_sparse: bool = False) -> Dict[str, Any]:
        """Generate embedding for the given text."""
        # Try to load the requested model
        if not await self._ensure_model_loaded(tier):
            # Try to find an alternative tier, preferring EMBEDDING tier
            for t in [ModelTier.EMBEDDING, ModelTier.LIGHT, ModelTier.MEDIUM, ModelTier.HEAVY]:
                if await self._ensure_model_loaded(t):
                    logger.warning(f"Model tier {tier.value} not available for embeddings, using {t.value}")
                    tier = t
                    break
            else:
                raise ValueError(f"No models available for embeddings. Requested tier: {tier.value}")
        
        model = self.models[tier]
        
        # Use the embedder if available
        if hasattr(model, 'embedder') and model.embedder is not None:
            result = await model.embedder.embed(text, embedding_type, return_sparse)
            return result.to_dict()
        else:
            # Fallback for models without dedicated embedder
            raise NotImplementedError(f"Embedding generation not supported for {tier.value} model. Please configure an embedding model.")
        
    async def rerank(self, query: str, documents: List[str], tier: ModelTier) -> List[float]:
        """Rerank documents based on query relevance."""
        # Try to load the requested model
        if not await self._ensure_model_loaded(tier):
            # Try to find an alternative tier, preferring RERANKER tier
            for t in [ModelTier.RERANKER, ModelTier.EMBEDDING, ModelTier.LIGHT]:
                if await self._ensure_model_loaded(t):
                    logger.warning(f"Model tier {tier.value} not available for reranking, using {t.value}")
                    tier = t
                    break
            else:
                raise ValueError(f"No models available for reranking. Requested tier: {tier.value}")
        
        model = self.models[tier]
        
        # Use the reranker if available
        if hasattr(model, 'reranker') and model.reranker is not None:
            result = await model.reranker.rerank(query, documents)
            return result.scores
        else:
            # Fallback: try to use embedder for similarity-based reranking
            if hasattr(model, 'embedder') and model.embedder is not None:
                # Create an embedding-based reranker
                reranker = create_reranker(model.model, self.configs[tier], model.embedder)
                result = await reranker.rerank(query, documents)
                return result.scores
            else:
                raise NotImplementedError(f"Reranking not supported for {tier.value} model. Please configure a reranker model.")
        
    def get_available_models(self, tier: Optional[ModelTier] = None) -> List[str]:
        """Get list of available model IDs (configured models, not necessarily loaded)."""
        if tier:
            if tier in self.configs:
                # Return model name based on path
                config = self.configs[tier]
                model_name = Path(config.path).stem
                loaded_suffix = " (loaded)" if tier in self.models else ""
                return [f"{tier.value}/{model_name}{loaded_suffix}"]
            return []
            
        # Return all configured models
        models = []
        for t, config in self.configs.items():
            model_name = Path(config.path).stem
            loaded_suffix = " (loaded)" if t in self.models else ""
            models.append(f"{t.value}/{model_name}{loaded_suffix}")
        return models
        
    def get_model_config(self, tier: ModelTier) -> Optional[ModelConfig]:
        """Get configuration for a model tier."""
        return self.configs.get(tier)
        
    async def cleanup(self):
        """Cleanup all models."""
        for wrapper in self.models.values():
            wrapper.cleanup()
        self.models.clear()
        self.configs.clear()