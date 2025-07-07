"""
Reranker model implementations for document reranking.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional, Dict
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking."""
    scores: List[float]
    indices: Optional[List[int]] = None  # Original indices if reordered
    
    def get_sorted_results(self, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float, str]]:
        """Get documents sorted by score."""
        # Create list of (index, score, document)
        results = [(i, score, doc) for i, (score, doc) in enumerate(zip(self.scores, documents))]
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k if specified
        if top_k is not None:
            results = results[:top_k]
            
        return results


class BaseReranker(ABC):
    """Abstract base class for reranking models."""
    
    def __init__(self, model: Any, config: Any):
        self.model = model
        self.config = config
        
    @abstractmethod
    async def rerank(self, query: str, documents: List[str]) -> RerankResult:
        """Rerank documents based on relevance to query."""
        pass
        
    @abstractmethod
    def get_model_type(self) -> str:
        """Get the type of reranker model."""
        pass


class BGEReranker(BaseReranker):
    """BGE-reranker specific implementation using cross-encoder architecture."""
    
    def get_model_type(self) -> str:
        return "bge-reranker"
        
    async def rerank(self, query: str, documents: List[str]) -> RerankResult:
        """Rerank documents using BGE cross-encoder approach."""
        scores = []
        
        loop = asyncio.get_event_loop()
        
        def _rerank_batch():
            batch_scores = []
            
            for doc in documents:
                # BGE reranker expects format: [CLS] query [SEP] document [SEP]
                # For GGUF models, we format it as a prompt
                prompt = f"Query: {query}\nDocument: {doc}\nRelevance:"
                
                try:
                    # Generate relevance score
                    # BGE reranker is trained to output a single score
                    result = self.model(
                        prompt,
                        max_tokens=1,
                        temperature=0.01,  # Very low temperature for consistency
                        echo=False,
                        logprobs=5  # Get logprobs for scoring
                    )
                    
                    # Extract score from model output
                    score = self._extract_score(result)
                    batch_scores.append(score)
                    
                except Exception as e:
                    logger.error(f"Error reranking document: {e}")
                    batch_scores.append(0.0)
                    
            return batch_scores
        
        scores = await loop.run_in_executor(None, _rerank_batch)
        
        return RerankResult(scores=scores)
    
    def _extract_score(self, result: Dict) -> float:
        """Extract relevance score from model output."""
        # BGE reranker typically outputs a score
        # We'll use logprobs as a proxy for confidence
        
        if "choices" in result and result["choices"]:
            choice = result["choices"][0]
            
            # Try to get logprobs
            if "logprobs" in choice and choice["logprobs"]:
                logprobs = choice["logprobs"]
                
                # Use average log probability as score
                if "token_logprobs" in logprobs and logprobs["token_logprobs"]:
                    token_logprobs = [lp for lp in logprobs["token_logprobs"] if lp is not None]
                    if token_logprobs:
                        # Convert log prob to probability and use as score
                        avg_logprob = sum(token_logprobs) / len(token_logprobs)
                        return float(avg_logprob)
                        
        # Default score if we can't extract from logprobs
        return 0.5


class SimpleReranker(BaseReranker):
    """Simple reranker using similarity scoring."""
    
    def get_model_type(self) -> str:
        return "simple-reranker"
        
    async def rerank(self, query: str, documents: List[str]) -> RerankResult:
        """Simple reranking using model perplexity or generation probability."""
        scores = []
        
        loop = asyncio.get_event_loop()
        
        def _rerank_batch():
            batch_scores = []
            
            for doc in documents:
                # Format as a question-answering prompt
                prompt = f"Question: {query}\nAnswer: {doc}\nIs this answer relevant? (yes/no):"
                
                try:
                    result = self.model(
                        prompt,
                        max_tokens=5,
                        temperature=0.1,
                        echo=False
                    )
                    
                    # Simple scoring based on output
                    text = result["choices"][0]["text"].lower().strip()
                    if "yes" in text:
                        score = 0.9
                    elif "no" in text:
                        score = 0.1
                    else:
                        score = 0.5
                        
                    batch_scores.append(score)
                    
                except Exception as e:
                    logger.error(f"Error reranking document: {e}")
                    batch_scores.append(0.0)
                    
            return batch_scores
        
        scores = await loop.run_in_executor(None, _rerank_batch)
        
        return RerankResult(scores=scores)


class EmbeddingReranker(BaseReranker):
    """Reranker using embedding similarity (for models without cross-encoder support)."""
    
    def __init__(self, model: Any, config: Any, embedder: Any):
        super().__init__(model, config)
        self.embedder = embedder
        
    def get_model_type(self) -> str:
        return "embedding-reranker"
        
    async def rerank(self, query: str, documents: List[str]) -> RerankResult:
        """Rerank using cosine similarity of embeddings."""
        # Get query embedding
        query_result = await self.embedder.embed(query)
        query_embedding = query_result.dense
        
        if query_embedding is None:
            raise ValueError("Failed to generate query embedding")
        
        # Get document embeddings
        doc_results = await self.embedder.embed(documents)
        
        scores = []
        for doc_result in doc_results:
            if doc_result.dense is None:
                scores.append(0.0)
                continue
                
            # Calculate cosine similarity
            score = self._cosine_similarity(query_embedding, doc_result.dense)
            scores.append(score)
            
        return RerankResult(scores=scores)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)


def create_reranker(model: Any, config: Any, embedder: Optional[Any] = None) -> BaseReranker:
    """Factory function to create appropriate reranker based on model type."""
    model_path = config.path.lower()
    
    # Detect model type based on path or config
    if "bge-reranker" in model_path or "cross-encoder" in model_path:
        logger.info("Creating BGE reranker")
        return BGEReranker(model, config)
    elif embedder is not None:
        logger.info("Creating embedding-based reranker")
        return EmbeddingReranker(model, config, embedder)
    else:
        logger.info("Creating simple reranker")
        return SimpleReranker(model, config)