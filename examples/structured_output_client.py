#!/usr/bin/env python3
"""
Example client for LLM Service structured output.

This demonstrates how to use the structured output feature to get clean JSON responses.
"""
import asyncio
import json
import httpx
from typing import Dict, Any, Optional, List
from pathlib import Path


class StructuredLLMClient:
    """Client for LLM Service with structured output support."""
    
    def __init__(self, base_url: Optional[str] = None):
        """Initialize client with automatic port discovery."""
        if base_url:
            self.base_url = base_url
        else:
            # Try to read port from .port file
            port_file = Path(".port")
            if port_file.exists():
                port = int(port_file.read_text().strip())
                self.base_url = f"http://localhost:{port}"
            else:
                self.base_url = "http://localhost:8000"
        
        print(f"Using LLM Service at: {self.base_url}")
    
    async def generate_json(
        self,
        prompt: str,
        model: str = "light",
        temperature: float = 0.1,
        max_tokens: int = 200,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output from a prompt.
        
        Args:
            prompt: The user prompt
            model: Model tier to use (light, medium, heavy)
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt for context
            
        Returns:
            Parsed JSON object
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        request = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=request,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Error {response.status_code}: {response.text}")
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                print(f"Raw content: {content}")
                raise
    
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract structured entities from text."""
        prompt = f"""Extract entities from this text and return as JSON with fields:
- people (array of names)
- organizations (array of names)
- locations (array of names)
- dates (array of dates)
- summary (brief summary)

Text: {text}"""
        
        return await self.generate_json(prompt, temperature=0.0)
    
    async def generate_product(self, product_type: str) -> Dict[str, Any]:
        """Generate a product listing."""
        prompt = f"""Generate a detailed product listing for a {product_type}.
Include: name, description, price, currency, inStock, category, features (array), specifications (object)"""
        
        return await self.generate_json(prompt, temperature=0.3)
    
    async def create_persona(self) -> Dict[str, Any]:
        """Generate a fictional person profile."""
        prompt = """Create a detailed persona with:
- name, age, occupation, location
- personality (object with traits)
- interests (array)
- background (brief text)
- goals (array of strings)"""
        
        return await self.generate_json(prompt, temperature=0.5)


# Example usage functions
async def example_basic_usage():
    """Basic usage example."""
    client = StructuredLLMClient()
    
    print("\n=== Basic JSON Generation ===")
    result = await client.generate_json(
        "Create a recipe for chocolate chip cookies with ingredients and steps"
    )
    print(json.dumps(result, indent=2))


async def example_entity_extraction():
    """Entity extraction example."""
    client = StructuredLLMClient()
    
    print("\n=== Entity Extraction ===")
    text = """
    Apple Inc. announced today that Tim Cook will visit Tokyo next month
    to meet with developers. The company, founded by Steve Jobs in 1976,
    continues to expand its presence in Japan.
    """
    
    entities = await client.extract_entities(text)
    print(json.dumps(entities, indent=2))


async def example_product_generation():
    """Product generation example."""
    client = StructuredLLMClient()
    
    print("\n=== Product Generation ===")
    products = []
    
    for product_type in ["wireless headphones", "smart watch", "portable speaker"]:
        product = await client.generate_product(product_type)
        products.append(product)
        print(f"\n{product_type.title()}:")
        print(json.dumps(product, indent=2))
    
    return products


async def example_batch_processing():
    """Batch processing example."""
    client = StructuredLLMClient()
    
    print("\n=== Batch Processing ===")
    
    prompts = [
        "Generate a user profile for a software developer",
        "Create a job posting for a data scientist position",
        "Design a course curriculum for Python programming"
    ]
    
    tasks = [client.generate_json(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    
    for prompt, result in zip(prompts, results):
        print(f"\nPrompt: {prompt[:50]}...")
        print(json.dumps(result, indent=2))


async def example_different_models():
    """Compare outputs from different models."""
    client = StructuredLLMClient()
    
    print("\n=== Model Comparison ===")
    prompt = "Generate a startup company profile with name, mission, and team"
    
    for model in ["light", "medium"]:
        print(f"\n--- {model.upper()} Model ---")
        try:
            result = await client.generate_json(prompt, model=model)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error with {model} model: {e}")


async def main():
    """Run all examples."""
    print("LLM Service Structured Output Examples")
    print("=" * 50)
    
    try:
        # Test connection
        client = StructuredLLMClient()
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(f"{client.base_url}/health")
            if response.status_code != 200:
                print("Error: LLM Service is not running!")
                print("Start it with: ./start_service.sh")
                return
        
        # Run examples
        await example_basic_usage()
        await example_entity_extraction()
        await example_product_generation()
        await example_batch_processing()
        await example_different_models()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure the LLM Service is running:")
        print("  ./start_service.sh")


if __name__ == "__main__":
    asyncio.run(main())