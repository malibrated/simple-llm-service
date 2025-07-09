#!/usr/bin/env python3
"""
LLM Service Client Library

A simple Python client library for interacting with the LLM Service API.
This can be used as a standalone module or copied into your project.

Usage:
    from llm_service_client import LLMServiceClient
    
    client = LLMServiceClient()
    response = await client.chat("Hello, how are you?")
"""

import httpx
import asyncio
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, AsyncIterator
from dataclasses import dataclass
from enum import Enum


class ModelTier(str, Enum):
    """Available model tiers."""
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    EMBEDDING = "embedding"
    RERANKER = "reranker"


@dataclass
class ChatMessage:
    """Chat message format."""
    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class EmbeddingResult:
    """Embedding result with optional sparse representation."""
    embedding: List[float]
    sparse_embedding: Optional[Dict[str, List[float]]] = None


class LLMServiceClient:
    """
    Client for interacting with the LLM Service API.
    
    Features:
    - Automatic port discovery
    - Type-safe methods
    - Async/await support
    - Error handling
    - Streaming support
    """
    
    def __init__(self, 
                 service_dir: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: float = 30.0,
                 auto_start: bool = False):
        """
        Initialize the client.
        
        Args:
            service_dir: Directory containing the .port file (default: current directory)
            base_url: Override the base URL (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
            auto_start: Automatically start the service if not running
        """
        self.service_dir = Path(service_dir or ".")
        self.base_url = base_url or self._discover_service_url()
        self.timeout = timeout
        self.auto_start = auto_start
        
        if auto_start:
            # Ensure service is running
            asyncio.create_task(self._ensure_service_running())
    
    def _discover_service_url(self) -> str:
        """Discover the service URL from the .port file."""
        port_file = self.service_dir / ".port"
        if port_file.exists():
            try:
                port = int(port_file.read_text().strip())
                return f"http://localhost:{port}"
            except:
                pass
        return "http://localhost:8000"
    
    async def _ensure_service_running(self):
        """Ensure the service is running, starting it if necessary."""
        if not await self.health_check():
            await self.start_service()
    
    async def start_service(self, wait_timeout: int = 30):
        """
        Start the LLM service if it's not running.
        
        Args:
            wait_timeout: Maximum seconds to wait for service to start
            
        Raises:
            RuntimeError: If service fails to start
        """
        import subprocess
        import time
        
        if await self.health_check():
            return  # Already running
        
        print("Starting LLM Service...")
        
        start_script = self.service_dir / "start_service.sh"
        if not start_script.exists():
            raise RuntimeError(f"start_service.sh not found in {self.service_dir}")
        
        if not start_script.is_file() or not os.access(start_script, os.X_OK):
            raise RuntimeError(f"start_service.sh is not executable. Run: chmod +x {start_script}")
        
        try:
            process = subprocess.Popen(
                [str(start_script)],
                cwd=str(self.service_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for service to start
            for i in range(wait_timeout):
                await asyncio.sleep(1)
                
                # Re-discover URL in case port changed
                self.base_url = self._discover_service_url()
                
                if await self.health_check():
                    print("Service started successfully!")
                    return
                
                # Check if process failed
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    error_msg = f"Service failed to start.\n"
                    if stdout:
                        error_msg += f"stdout: {stdout}\n"
                    if stderr:
                        error_msg += f"stderr: {stderr}"
                    raise RuntimeError(error_msg)
            
            # Timeout
            process.terminate()
            raise RuntimeError(f"Service failed to start within {wait_timeout} seconds")
            
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to start service: {e}")
    
    # === Chat Methods ===
    
    async def chat(self,
                   content: Union[str, List[ChatMessage]],
                   model: Union[str, ModelTier] = ModelTier.MEDIUM,
                   system_prompt: Optional[str] = None,
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None,
                   **kwargs) -> str:
        """
        Send a chat completion request.
        
        Args:
            content: Message string or list of ChatMessage objects
            model: Model tier or specific model name
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (top_p, top_k, seed, etc.)
            
        Returns:
            The assistant's response content
        """
        messages = self._prepare_messages(content, system_prompt)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": str(model),
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    
    async def chat_stream(self,
                          content: Union[str, List[ChatMessage]],
                          model: Union[str, ModelTier] = ModelTier.MEDIUM,
                          system_prompt: Optional[str] = None,
                          **kwargs) -> AsyncIterator[str]:
        """
        Stream a chat completion response.
        
        Yields:
            Response content chunks as they arrive
        """
        messages = self._prepare_messages(content, system_prompt)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": str(model),
                    "messages": messages,
                    "stream": True,
                    **kwargs
                }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "content" in chunk["choices"][0]["delta"]:
                                yield chunk["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError:
                            continue
    
    async def structured_chat(self,
                              content: str,
                              model: Union[str, ModelTier] = ModelTier.MEDIUM,
                              system_prompt: Optional[str] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Get structured JSON output from chat.
        
        Returns:
            Parsed JSON object
        """
        messages = self._prepare_messages(content, system_prompt)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": str(model),
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1,  # Lower temperature for consistency
                    **kwargs
                }
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
    
    # === Completion Methods (Legacy) ===
    
    async def complete(self,
                       prompt: str,
                       model: Union[str, ModelTier] = ModelTier.MEDIUM,
                       **kwargs) -> str:
        """
        Text completion (legacy format).
        
        Args:
            prompt: The prompt text
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            The completed text
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": str(model),
                    "prompt": prompt,
                    **kwargs
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"]
    
    # === Embedding Methods ===
    
    async def embed(self,
                    text: Union[str, List[str]],
                    model: Union[str, ModelTier] = ModelTier.EMBEDDING,
                    embedding_type: str = "dense",
                    return_sparse: bool = False) -> Union[List[float], List[List[float]], List[EmbeddingResult]]:
        """
        Generate embeddings for text.
        
        Args:
            text: Single text or list of texts
            model: Embedding model to use
            embedding_type: "dense", "sparse", or "colbert" (BGE-M3)
            return_sparse: Return both dense and sparse embeddings
            
        Returns:
            Single embedding, list of embeddings, or EmbeddingResult objects
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/embeddings",
                json={
                    "model": str(model),
                    "input": texts,
                    "embedding_type": embedding_type,
                    "return_sparse": return_sparse
                }
            )
            response.raise_for_status()
            result = response.json()
            
            if return_sparse:
                embeddings = [
                    EmbeddingResult(
                        embedding=item["embedding"],
                        sparse_embedding=item.get("sparse_embedding")
                    )
                    for item in result["data"]
                ]
            else:
                embeddings = [item["embedding"] for item in result["data"]]
            
            return embeddings[0] if is_single else embeddings
    
    # === Reranking Methods ===
    
    async def rerank(self,
                     query: str,
                     documents: List[str],
                     model: Union[str, ModelTier] = ModelTier.RERANKER,
                     top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            model: Reranker model to use
            top_n: Return only top N documents
            
        Returns:
            List of documents with scores, sorted by relevance
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/rerank",
                json={
                    "model": str(model),
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                    "return_documents": True
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["data"]
    
    # === Utility Methods ===
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            result = response.json()
            return result["data"]
    
    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except:
            return False
    
    def _prepare_messages(self,
                          content: Union[str, List[ChatMessage]],
                          system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Prepare messages for chat endpoint."""
        if isinstance(content, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages = [{"role": msg.role, "content": msg.content} for msg in content]
            if system_prompt and not any(msg["role"] == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})
        
        return messages


# Convenience functions for quick usage
async def quick_chat(prompt: str, model: str = "medium", auto_start: bool = True, **kwargs) -> str:
    """Quick chat without creating a client instance."""
    client = LLMServiceClient(auto_start=auto_start)
    if auto_start:
        await client._ensure_service_running()
    return await client.chat(prompt, model=model, **kwargs)


async def quick_json(prompt: str, model: str = "medium", auto_start: bool = True, **kwargs) -> Dict[str, Any]:
    """Quick structured output without creating a client instance."""
    client = LLMServiceClient(auto_start=auto_start)
    if auto_start:
        await client._ensure_service_running()
    return await client.structured_chat(prompt, model=model, **kwargs)


# Example usage
if __name__ == "__main__":
    async def example():
        # Create client with auto-start enabled
        client = LLMServiceClient(auto_start=True)
        
        # Ensure service is running (will start if needed)
        await client._ensure_service_running()
        
        # Simple chat
        response = await client.chat("What is the capital of France?")
        print(f"Chat: {response}")
        
        # Structured output
        data = await client.structured_chat(
            "List 3 programming languages with their creation year"
        )
        print(f"Structured: {json.dumps(data, indent=2)}")
        
        # Embeddings
        try:
            embedding = await client.embed("Hello world")
            print(f"Embedding dimension: {len(embedding)}")
        except Exception as e:
            print(f"Embedding skipped (model may not be configured): {e}")
        
        # Stream
        print("Stream: ", end="")
        async for chunk in client.chat_stream("Tell me a joke"):
            print(chunk, end="", flush=True)
        print()
    
    asyncio.run(example())