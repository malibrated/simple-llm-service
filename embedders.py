"""
Embedding model implementations with support for various embedding types.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    dense: Optional[List[float]] = None
    sparse: Optional[Dict[int, float]] = None  # token_id -> weight
    colbert: Optional[List[List[float]]] = None  # multi-vector embeddings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {}
        if self.dense is not None:
            result["embedding"] = self.dense
            result["dimension"] = len(self.dense)
        if self.sparse is not None:
            result["sparse_embedding"] = self.sparse
        if self.colbert is not None:
            result["colbert_embedding"] = self.colbert
        return result


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self, model: Any, config: Any):
        self.model = model
        self.config = config
        
    @abstractmethod
    async def embed(self, 
                   text: Union[str, List[str]], 
                   embedding_type: str = "dense",
                   return_sparse: bool = False) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """Generate embeddings for text(s)."""
        pass
        
    @abstractmethod
    def supports_sparse(self) -> bool:
        """Check if this embedder supports sparse embeddings."""
        pass
        
    @abstractmethod
    def supports_colbert(self) -> bool:
        """Check if this embedder supports ColBERT embeddings."""
        pass


class BGEM3Embedder(BaseEmbedder):
    """BGE-M3 specific embedder with dense, sparse, and ColBERT support."""
    
    def supports_sparse(self) -> bool:
        return True
        
    def supports_colbert(self) -> bool:
        return True
        
    async def embed(self,
                   text: Union[str, List[str]],
                   embedding_type: str = "dense", 
                   return_sparse: bool = False) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """Generate BGE-M3 embeddings with support for multiple types."""
        # Handle single text vs list
        texts = [text] if isinstance(text, str) else text
        single_input = isinstance(text, str)
        
        results = []
        
        for txt in texts:
            result = EmbeddingResult()
            
            # Generate embeddings based on type
            if embedding_type in ["dense", "colbert"] or return_sparse:
                # BGE-M3 can generate dense embeddings
                if hasattr(self.model, 'embed'):
                    loop = asyncio.get_event_loop()
                    
                    def _embed():
                        # Get dense embedding
                        dense_embedding = self.model.embed(txt)
                        return dense_embedding
                    
                    dense_embedding = await loop.run_in_executor(None, _embed)
                    result.dense = dense_embedding
            
            # Generate sparse embeddings if requested
            if embedding_type == "sparse" or return_sparse:
                sparse_embedding = await self._generate_sparse_embedding(txt)
                result.sparse = sparse_embedding
                
            # ColBERT embeddings would require special handling
            if embedding_type == "colbert":
                # For now, we'll use the dense embedding
                # Real ColBERT would generate one embedding per token
                if result.dense:
                    # Mock: split into multiple vectors
                    # In practice, ColBERT generates token-level embeddings
                    result.colbert = [result.dense]  # Simplified
                    
            results.append(result)
        
        return results[0] if single_input else results
    
    async def _generate_sparse_embedding(self, text: str) -> Dict[int, float]:
        """Generate sparse embeddings for BGE-M3."""
        loop = asyncio.get_event_loop()
        
        def _sparse_embed():
            # BGE-M3 uses learned sparse embeddings
            # This is a simplified version - real BGE-M3 has special sparse tokens
            if hasattr(self.model, 'tokenize'):
                tokens = self.model.tokenize(text.encode('utf-8'))
                
                # Create sparse representation
                # In real BGE-M3, weights are learned through training
                sparse_dict = {}
                for token_id in tokens:
                    # Skip special tokens
                    if token_id < 50000:  # Approximate vocabulary size
                        if token_id not in sparse_dict:
                            sparse_dict[token_id] = 0
                        sparse_dict[token_id] += 1.0
                
                # Normalize weights (simplified - BGE-M3 uses learned weights)
                total = sum(sparse_dict.values())
                if total > 0:
                    sparse_dict = {k: v/total for k, v in sparse_dict.items()}
                    
                return sparse_dict
            return {}
            
        return await loop.run_in_executor(None, _sparse_embed)


class SimpleEmbedder(BaseEmbedder):
    """Simple embedder for general models that only support dense embeddings."""
    
    def supports_sparse(self) -> bool:
        return False
        
    def supports_colbert(self) -> bool:
        return False
        
    async def embed(self,
                   text: Union[str, List[str]],
                   embedding_type: str = "dense",
                   return_sparse: bool = False) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """Generate simple dense embeddings."""
        if embedding_type != "dense":
            logger.warning(f"SimpleEmbedder only supports dense embeddings, ignoring type: {embedding_type}")
            
        # Handle single text vs list
        texts = [text] if isinstance(text, str) else text
        single_input = isinstance(text, str)
        
        results = []
        
        for txt in texts:
            result = EmbeddingResult()
            
            if hasattr(self.model, 'embed'):
                loop = asyncio.get_event_loop()
                
                def _embed():
                    return self.model.embed(txt)
                
                dense_embedding = await loop.run_in_executor(None, _embed)
                result.dense = dense_embedding
            else:
                raise NotImplementedError("Model does not support embeddings")
                
            results.append(result)
        
        return results[0] if single_input else results


def create_embedder(model: Any, config: Any) -> BaseEmbedder:
    """Factory function to create appropriate embedder based on model type."""
    model_path = config.path.lower()
    
    # Detect model type based on path or config
    if "bge-m3" in model_path:
        logger.info("Creating BGE-M3 embedder")
        return BGEM3Embedder(model, config)
    else:
        logger.info("Creating simple embedder")
        return SimpleEmbedder(model, config)