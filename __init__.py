"""
LLM Service - OpenAI-compatible API for local LLMs.
"""

__version__ = "1.0.0"

# Only import when used as a package
try:
    from .model_manager import ModelManager, ModelTier, ModelConfig, InferenceBackend
    from .cache import ResponseCache, SemanticCache
    from .embedders import create_embedder, BaseEmbedder
    from .rerankers import create_reranker, BaseReranker
    
    __all__ = [
        "ModelManager",
        "ModelTier", 
        "ModelConfig",
        "InferenceBackend",
        "ResponseCache",
        "SemanticCache",
        "create_embedder",
        "BaseEmbedder",
        "create_reranker",
        "BaseReranker",
    ]
except ImportError:
    # When running scripts directly, imports will fail
    pass