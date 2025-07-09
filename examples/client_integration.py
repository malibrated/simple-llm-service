#!/usr/bin/env python3
"""
Client Integration Examples for LLM Service

This file demonstrates various ways to integrate with the LLM Service API.
"""

import httpx
import asyncio
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import time


class LLMServiceClient:
    """A simple client for the LLM Service with automatic port discovery."""
    
    def __init__(self, service_dir: str = "."):
        """Initialize client with service directory path."""
        self.service_dir = Path(service_dir)
        self.base_url = self._discover_service_url()
        
    def _discover_service_url(self) -> str:
        """Discover the service URL from the .port file."""
        port_file = self.service_dir / ".port"
        if port_file.exists():
            port = int(port_file.read_text().strip())
            return f"http://localhost:{port}"
        # Fallback to default
        return "http://localhost:8000"
    
    async def chat(self, 
                   messages: List[Dict[str, str]], 
                   model: str = "medium",
                   **kwargs) -> str:
        """Send a chat completion request."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    **kwargs
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    
    async def chat_stream(self,
                          messages: List[Dict[str, str]],
                          model: str = "medium",
                          **kwargs):
        """Stream a chat completion response."""
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model,
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
                        chunk = json.loads(data)
                        if "content" in chunk["choices"][0]["delta"]:
                            yield chunk["choices"][0]["delta"]["content"]
    
    async def structured_output(self,
                                prompt: str,
                                model: str = "medium",
                                **kwargs) -> Dict[str, Any]:
        """Get structured JSON output."""
        response = await self.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            response_format={"type": "json_object"},
            temperature=0.1,
            **kwargs
        )
        return json.loads(response)
    
    async def embeddings(self,
                         texts: List[str],
                         model: str = "embedding",
                         embedding_type: str = "dense") -> List[List[float]]:
        """Generate embeddings for texts."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/embeddings",
                json={
                    "model": model,
                    "input": texts,
                    "embedding_type": embedding_type
                }
            )
            response.raise_for_status()
            result = response.json()
            return [item["embedding"] for item in result["data"]]
    
    async def rerank(self,
                     query: str,
                     documents: List[str],
                     model: str = "reranker",
                     top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank documents by relevance to query."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/rerank",
                json={
                    "model": model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                    "return_documents": True
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["data"]
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            result = response.json()
            return result["data"]
    
    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except:
            return False


# Example usage functions
async def basic_chat_example():
    """Basic chat completion example."""
    print("=== Basic Chat Example ===")
    client = LLMServiceClient()
    
    response = await client.chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the three primary colors?"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    print(f"Response: {response}")


async def streaming_example():
    """Streaming chat completion example."""
    print("\n=== Streaming Example ===")
    client = LLMServiceClient()
    
    print("Assistant: ", end="", flush=True)
    async for chunk in client.chat_stream(
        messages=[{"role": "user", "content": "Write a haiku about programming"}],
        temperature=0.8
    ):
        print(chunk, end="", flush=True)
    print()


async def structured_output_example():
    """Structured JSON output example."""
    print("\n=== Structured Output Example ===")
    client = LLMServiceClient()
    
    # Generate a product listing
    product = await client.structured_output(
        prompt="Generate a product listing for a laptop with name, price, specs (CPU, RAM, storage), and description",
        model="medium"
    )
    print(f"Product: {json.dumps(product, indent=2)}")
    
    # Extract information
    entities = await client.structured_output(
        prompt="Extract entities from: 'Apple CEO Tim Cook announced the new MacBook Pro in Cupertino.' Return as {people: [], companies: [], products: [], locations: []}",
        model="medium"
    )
    print(f"Entities: {json.dumps(entities, indent=2)}")


async def embedding_similarity_example():
    """Embedding and similarity example."""
    print("\n=== Embedding Similarity Example ===")
    client = LLMServiceClient()
    
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks",
        "I love eating pizza",
        "Natural language processing is fascinating"
    ]
    
    embeddings = await client.embeddings(texts)
    
    # Calculate cosine similarity
    import numpy as np
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Find most similar to first text
    print(f"Query: '{texts[0]}'")
    similarities = []
    for i in range(1, len(texts)):
        sim = cosine_similarity(embeddings[0], embeddings[i])
        similarities.append((sim, texts[i]))
    
    similarities.sort(reverse=True)
    print("\nMost similar texts:")
    for sim, text in similarities:
        print(f"  {sim:.3f}: {text}")


async def reranking_example():
    """Document reranking example."""
    print("\n=== Reranking Example ===")
    client = LLMServiceClient()
    
    query = "How to implement a REST API in Python?"
    documents = [
        "FastAPI is a modern web framework for building APIs with Python",
        "The weather today is sunny and warm",
        "Flask is a lightweight WSGI web application framework",
        "REST stands for Representational State Transfer",
        "I enjoy cooking Italian food on weekends",
        "Django REST framework is powerful for building Web APIs"
    ]
    
    ranked = await client.rerank(query, documents, top_n=3)
    
    print(f"Query: '{query}'")
    print("\nTop relevant documents:")
    for item in ranked:
        print(f"  Score {item['score']:.3f}: {item['document']}")


async def model_selection_example():
    """Model tier selection example."""
    print("\n=== Model Selection Example ===")
    client = LLMServiceClient()
    
    # List available models
    models = await client.list_models()
    print("Available models:")
    for model in models:
        status = " (loaded)" if "(loaded)" in model["id"] else ""
        print(f"  - {model['id']}{status}")
    
    # Use different models for different tasks
    tasks = [
        ("light", "What is 2+2?"),
        ("medium", "Explain quantum computing in one sentence"),
        ("heavy", "Write a detailed analysis of climate change impacts")
    ]
    
    for model, prompt in tasks:
        print(f"\nUsing {model} model:")
        try:
            start = time.time()
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=100
            )
            elapsed = time.time() - start
            print(f"Response ({elapsed:.2f}s): {response[:100]}...")
        except Exception as e:
            print(f"Error: {e}")


async def error_handling_example():
    """Error handling example."""
    print("\n=== Error Handling Example ===")
    client = LLMServiceClient()
    
    # Check health
    is_healthy = await client.health_check()
    print(f"Service healthy: {is_healthy}")
    
    # Try invalid model
    try:
        await client.chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="invalid_model"
        )
    except httpx.HTTPStatusError as e:
        print(f"Expected error for invalid model: {e.response.status_code}")
        error_detail = e.response.json()
        print(f"Error details: {error_detail}")


async def main():
    """Run all examples."""
    examples = [
        basic_chat_example,
        streaming_example,
        structured_output_example,
        embedding_similarity_example,
        reranking_example,
        model_selection_example,
        error_handling_example
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
        print()


if __name__ == "__main__":
    # Check if service is running
    client = LLMServiceClient()
    if not asyncio.run(client.health_check()):
        print("LLM Service is not running. Starting service...")
        
        # Start the service
        import subprocess
        import time
        
        try:
            # Start the service in the background
            process = subprocess.Popen(
                ["./start_service.sh"],
                cwd=client.service_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for service to start (check every second for up to 30 seconds)
            for i in range(30):
                time.sleep(1)
                if asyncio.run(client.health_check()):
                    print("Service started successfully!")
                    break
                # Check if process failed
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    print(f"Failed to start service:")
                    if stdout:
                        print(f"stdout: {stdout}")
                    if stderr:
                        print(f"stderr: {stderr}")
                    exit(1)
            else:
                print("Service failed to start within 30 seconds")
                process.terminate()
                exit(1)
                
        except FileNotFoundError:
            print("Error: start_service.sh not found in service directory")
            print(f"Please ensure you're running from the service directory: {client.service_dir}")
            exit(1)
        except PermissionError:
            print("Error: start_service.sh is not executable")
            print("Please run: chmod +x start_service.sh")
            exit(1)
        except Exception as e:
            print(f"Error starting service: {e}")
            exit(1)
    
    # Run examples
    asyncio.run(main())