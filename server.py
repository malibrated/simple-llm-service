#!/usr/bin/env python3
"""
LLM Service with OpenAI-compatible REST API.
Supports both llama.cpp and MLX models with configurable parameters.
"""
import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Load environment variables
load_dotenv()

# Helper to parse env vars with inline comments
def _parse_env(key: str, default: str = "") -> str:
    """Parse environment variable, stripping inline comments."""
    value = os.getenv(key, default)
    if '#' in value:
        # Strip inline comments
        value = value.split('#')[0].strip()
    return value

# Configure logging
logging.basicConfig(
    level=_parse_env("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Request/Response Models (OpenAI Compatible)
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    n: Optional[int] = Field(default=1, ge=1, le=10)
    user: Optional[str] = None
    
    # Additional parameters for fine-grained control
    repeat_penalty: Optional[float] = Field(default=None, ge=0.0)
    seed: Optional[int] = None
    grammar: Optional[str] = None  # JSON grammar for structured output
    response_format: Optional[Dict[str, Any]] = None  # For JSON mode


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llmservice"
    permission: List[Dict] = Field(default_factory=list)
    root: str
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# Completion API Models (for backwards compatibility)
class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    n: Optional[int] = Field(default=1, ge=1, le=10)
    echo: Optional[bool] = False
    
    # Additional parameters
    top_k: Optional[int] = Field(default=None, ge=1)
    repeat_penalty: Optional[float] = Field(default=None, ge=0.0)
    seed: Optional[int] = None
    grammar: Optional[str] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


# Embedding Models
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    # BGE-M3 specific options
    embedding_type: Optional[Literal["dense", "sparse", "colbert"]] = "dense"
    return_sparse: Optional[bool] = False  # Return sparse embeddings along with dense


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]
    # Optional sparse embedding data for BGE-M3
    sparse_embedding: Optional[Dict[int, float]] = None  # {token_id: weight}


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage


# Reranking Models
class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_k: Optional[int] = None  # Return top K documents
    return_documents: Optional[bool] = True  # Return documents with scores


class RerankResult(BaseModel):
    index: int
    score: float
    document: Optional[str] = None


class RerankResponse(BaseModel):
    object: str = "list"
    data: List[RerankResult]
    model: str
    usage: Usage


# Model Management
from model_manager import ModelManager, ModelConfig, ModelTier
from cache import ResponseCache


class LLMService:
    """Main LLM Service with OpenAI-compatible API."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.cache = ResponseCache(enabled=_parse_env("ENABLE_CACHE", "true").lower() == "true")
        self.last_activity = time.time()
        self.shutdown_timeout = int(_parse_env("SHUTDOWN_TIMEOUT", "600"))  # 10 minutes default
        self._shutdown_task: Optional[asyncio.Task] = None
        
    async def startup(self):
        """Initialize service on startup."""
        logger.info("Starting LLM Service...")
        
        # Initialize model configurations (lazy loading enabled)
        await self.model_manager.initialize()
        
        # Start shutdown monitor
        if self.shutdown_timeout > 0:
            self._shutdown_task = asyncio.create_task(self._monitor_shutdown())
            
        logger.info("LLM Service ready")
        
    async def shutdown(self):
        """Cleanup on shutdown."""
        logger.info("Shutting down LLM Service...")
        
        if self._shutdown_task:
            self._shutdown_task.cancel()
            
        await self.model_manager.cleanup()
        self.cache.clear()
        
        logger.info("LLM Service stopped")
        
    async def _monitor_shutdown(self):
        """Monitor for inactivity and shutdown if needed."""
        while True:
            try:
                await asyncio.sleep(30)
                
                if time.time() - self.last_activity > self.shutdown_timeout:
                    logger.info(f"No activity for {self.shutdown_timeout}s, initiating shutdown...")
                    os.kill(os.getpid(), signal.SIGTERM)
                    break
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in shutdown monitor: {e}")
                
    def _update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
        
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion request (OpenAI compatible)."""
        self._update_activity()
        
        # Map model name to tier
        model_tier = self._get_model_tier(request.model)
        
        # Convert chat messages to prompt
        prompt = self._messages_to_prompt(request.messages)
        
        # Check cache
        cache_key = self.cache.get_key(prompt, request.model, request.dict(exclude={"messages", "user"}))
        cached = self.cache.get(cache_key)
        if cached:
            logger.info(f"Cache hit for model {request.model}")
            return ChatCompletionResponse(**cached)
            
        # Get model defaults
        model_config = self.model_manager.get_model_config(model_tier)
        
        # Prepare generation kwargs
        generation_params = {
            "temperature": request.temperature if request.temperature is not None else model_config.temperature,
            "top_p": request.top_p if request.top_p is not None else model_config.top_p,
            "top_k": request.top_k if request.top_k is not None else model_config.top_k,
            "max_tokens": request.max_tokens if request.max_tokens is not None else model_config.max_tokens,
            "repeat_penalty": request.repeat_penalty if request.repeat_penalty is not None else model_config.repeat_penalty,
            "stop": request.stop,
            "seed": request.seed,
        }
        
        # Handle grammar/structured output
        if request.grammar:
            generation_params["grammar"] = request.grammar
        elif request.response_format and request.response_format.get("type") == "json_object":
            # Convert to appropriate grammar for JSON output
            generation_params["grammar"] = self._get_json_grammar()
            
        # Generate response
        try:
            result = await self.model_manager.generate(
                model_tier=model_tier,
                prompt=prompt,
                **generation_params
            )
            
            # Build response
            response = ChatCompletionResponse(
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=ChatMessage(role="assistant", content=result["text"]),
                        finish_reason=result.get("finish_reason", "stop")
                    )
                ],
                usage=Usage(
                    prompt_tokens=result.get("prompt_tokens", 0),
                    completion_tokens=result.get("completion_tokens", 0),
                    total_tokens=result.get("total_tokens", 0)
                )
            )
            
            # Cache response
            self.cache.set(cache_key, response.dict())
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def completion(self, request: CompletionRequest) -> CompletionResponse:
        """Handle completion request (OpenAI compatible)."""
        self._update_activity()
        
        # Map model name to tier
        model_tier = self._get_model_tier(request.model)
        
        # Handle multiple prompts
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        choices = []
        
        for idx, prompt in enumerate(prompts[:request.n]):
            # Check cache
            cache_key = self.cache.get_key(prompt, request.model, request.dict(exclude={"prompt"}))
            cached = self.cache.get(cache_key)
            
            if cached:
                choices.append(CompletionChoice(**cached["choices"][0]))
                continue
                
            # Get model defaults
            model_config = self.model_manager.get_model_config(model_tier)
            
            # Prepare generation kwargs
            generation_params = {
                "temperature": request.temperature if request.temperature is not None else model_config.temperature,
                "top_p": request.top_p if request.top_p is not None else model_config.top_p,
                "top_k": request.top_k if request.top_k is not None else model_config.top_k,
                "max_tokens": request.max_tokens if request.max_tokens is not None else model_config.max_tokens,
                "repeat_penalty": request.repeat_penalty if request.repeat_penalty is not None else model_config.repeat_penalty,
                "stop": request.stop,
                "seed": request.seed,
            }
            
            if request.grammar:
                generation_params["grammar"] = request.grammar
                
            # Generate
            try:
                result = await self.model_manager.generate(
                    model_tier=model_tier,
                    prompt=prompt,
                    **generation_params
                )
                
                text = result["text"]
                if request.echo:
                    text = prompt + text
                    
                choice = CompletionChoice(
                    text=text,
                    index=idx,
                    finish_reason=result.get("finish_reason", "stop")
                )
                choices.append(choice)
                
                # Cache individual result
                self.cache.set(cache_key, {"choices": [choice.dict()]})
                
            except Exception as e:
                logger.error(f"Error in completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        # Build response
        total_tokens = sum(c.text.count(" ") * 1.3 for c in choices)  # Rough estimate
        response = CompletionResponse(
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=int(total_tokens * 0.3),
                completion_tokens=int(total_tokens * 0.7),
                total_tokens=int(total_tokens)
            )
        )
        
        return response
        
    async def embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for the given input."""
        # Update activity
        self.last_activity = time.time()
        
        # Ensure input is a list
        inputs = request.input if isinstance(request.input, list) else [request.input]
        
        # Get model tier
        tier = self._get_model_tier(request.model)
        
        # Generate embeddings
        embeddings_data = []
        total_tokens = 0
        
        for idx, text in enumerate(inputs):
            # Check cache first - include embedding type in cache key
            cache_key = f"emb:{tier.value}:{request.embedding_type}:{hash(text)}"
            cached = self.cache.get(cache_key)
            
            if cached:
                embedding = cached["embedding"]
                sparse_embedding = cached.get("sparse_embedding")
            else:
                try:
                    # Generate embedding with type specification
                    result = await self.model_manager.generate_embedding(
                        text=text,
                        tier=tier,
                        embedding_type=request.embedding_type,
                        return_sparse=request.return_sparse
                    )
                    embedding = result["embedding"]
                    sparse_embedding = result.get("sparse_embedding")
                    
                    # Cache result
                    cache_data = {"embedding": embedding}
                    if sparse_embedding:
                        cache_data["sparse_embedding"] = sparse_embedding
                    self.cache.set(cache_key, cache_data)
                    
                except Exception as e:
                    logger.error(f"Error generating embedding: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Handle dimension reduction if requested
            if request.dimensions and len(embedding) > request.dimensions:
                # Simple truncation - you might want more sophisticated dimensionality reduction
                embedding = embedding[:request.dimensions]
            
            embedding_data = EmbeddingData(
                index=idx,
                embedding=embedding
            )
            
            # Add sparse embedding if available
            if request.return_sparse and sparse_embedding:
                embedding_data.sparse_embedding = sparse_embedding
                
            embeddings_data.append(embedding_data)
            
            # Estimate tokens
            total_tokens += len(text.split()) * 1.3  # Rough estimate
        
        return EmbeddingResponse(
            data=embeddings_data,
            model=request.model,
            usage=Usage(
                prompt_tokens=int(total_tokens),
                completion_tokens=0,
                total_tokens=int(total_tokens)
            )
        )
        
    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """Rerank documents based on query relevance."""
        # Update activity
        self.last_activity = time.time()
        
        # Get model tier (prefer RERANKER, fallback to others)
        tier = self._get_model_tier(request.model)
        
        # Rerank documents
        results = []
        total_tokens = len(request.query.split()) * 1.3  # Query tokens
        
        # Get scores for each document
        scores = await self.model_manager.rerank(
            query=request.query,
            documents=request.documents,
            tier=tier
        )
        
        # Create results with scores
        for idx, (doc, score) in enumerate(zip(request.documents, scores)):
            results.append(RerankResult(
                index=idx,
                score=score,
                document=doc if request.return_documents else None
            ))
            total_tokens += len(doc.split()) * 1.3
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply top_k if specified
        if request.top_k is not None:
            results = results[:request.top_k]
        
        return RerankResponse(
            data=results,
            model=request.model,
            usage=Usage(
                prompt_tokens=int(total_tokens),
                completion_tokens=0,
                total_tokens=int(total_tokens)
            )
        )
        
    def list_models(self) -> ModelsResponse:
        """List available models."""
        models = []
        
        for tier in ModelTier:
            tier_models = self.model_manager.get_available_models(tier)
            for model_id in tier_models:
                models.append(ModelInfo(
                    id=model_id,
                    root=model_id,
                    owned_by=f"llmservice-{tier.value}"
                ))
                
        return ModelsResponse(data=models)
        
    def _get_model_tier(self, model_name: str) -> ModelTier:
        """Map model name to tier."""
        # Check if it's a direct tier name
        if model_name in [t.value for t in ModelTier]:
            return ModelTier(model_name)
            
        # Check configured mappings
        model_name_lower = model_name.lower()
        
        # Special handling for embedding models
        if any(x in model_name_lower for x in ["embed", "bge", "e5", "gte", "sentence"]):
            return ModelTier.EMBEDDING
        
        # Special handling for reranker models
        if any(x in model_name_lower for x in ["rerank", "ranker", "cross-encoder"]):
            return ModelTier.RERANKER
        
        # Default mappings based on model size
        if any(x in model_name_lower for x in ["0.5b", "1b", "tiny", "small"]):
            return ModelTier.LIGHT
        elif any(x in model_name_lower for x in ["3b", "4b", "7b", "medium"]):
            return ModelTier.MEDIUM
        elif any(x in model_name_lower for x in ["13b", "24b", "70b", "large", "heavy"]):
            return ModelTier.HEAVY
            
        # Default to medium
        return ModelTier.MEDIUM
        
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a single prompt."""
        # This is a simple implementation - you might want to use model-specific templates
        prompt_parts = []
        
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
                
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
        
    def _get_json_grammar(self) -> str:
        """Get a basic JSON grammar."""
        return '''
root ::= object
object ::= "{" ws members ws "}"
members ::= member ("," ws member)*
member ::= string ws ":" ws value
value ::= object | array | string | number | boolean | null
array ::= "[" ws elements ws "]"
elements ::= value ("," ws value)*
string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
boolean ::= "true" | "false"
null ::= "null"
ws ::= [ \\t\\n]*
'''


# Create FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    port = int(os.getenv("PORT", "8000"))
    port_file = Path(".port")
    
    # Write port to file
    try:
        port_file.write_text(str(port))
        logger.info(f"Written port {port} to {port_file}")
    except Exception as e:
        logger.error(f"Failed to write port file: {e}")
    
    await app.state.service.startup()
    yield
    # Shutdown
    await app.state.service.shutdown()
    
    # Remove port file
    try:
        if port_file.exists():
            port_file.unlink()
            logger.info(f"Removed port file {port_file}")
    except Exception as e:
        logger.error(f"Failed to remove port file: {e}")


# Create FastAPI app
app = FastAPI(
    title="LLM Service",
    description="OpenAI-compatible LLM service with support for multiple models and platforms",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
service = LLMService()
app.state.service = service


# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models."""
    return service.list_models()


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings for the given input."""
    return await service.embedding(request)


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """Rerank documents based on query relevance."""
    return await service.rerank(request)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Create chat completion."""
    return await service.chat_completion(request)


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """Create completion."""
    return await service.completion(request)


# Legacy endpoints for compatibility
@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def legacy_chat_completions(request: ChatCompletionRequest):
    """Legacy chat completion endpoint."""
    return await service.chat_completion(request)


@app.post("/completions", response_model=CompletionResponse)
async def legacy_completions(request: CompletionRequest):
    """Legacy completion endpoint."""
    return await service.completion(request)


if __name__ == "__main__":
    # Run with uvicorn
    port = int(_parse_env("PORT", "8000"))
    host = _parse_env("HOST", "0.0.0.0")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=_parse_env("RELOAD", "false").lower() == "true",
        log_level=_parse_env("LOG_LEVEL", "info").lower()
    )