#!/usr/bin/env python3
"""Debug MLX JSON generation issue"""
import sys
import json
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '.')

async def test_mlx_directly():
    """Test MLX Outlines generation directly"""
    try:
        from outlines import models, generate
        from mlx_lm import load as mlx_load
        from pydantic import BaseModel
        
        print("Testing MLX Outlines generation directly...")
        
        # Get MLX model path from env
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        model_path = os.getenv("MEDIUM_MODEL_PATH")
        if not model_path:
            print("No MEDIUM_MODEL_PATH found")
            return
            
        print(f"Loading model from: {model_path}")
        
        # Load model
        mlx_model, tokenizer = mlx_load(model_path)
        model = models.from_mlxlm(mlx_model, tokenizer)
        
        # Test 1: With flexible schema
        print("\n1. Testing with flexible schema (extra='allow'):")
        class FlexibleModel(BaseModel):
            class Config:
                extra = 'allow'
        
        generator = generate.json(model, FlexibleModel)
        result = generator("Generate a simple JSON object with name and value fields", max_tokens=50)
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        if hasattr(result, 'model_dump'):
            print(f"Dumped: {result.model_dump()}")
        
        # Test 2: With specific schema
        print("\n2. Testing with specific schema:")
        class SpecificModel(BaseModel):
            name: str
            value: int
        
        generator2 = generate.json(model, SpecificModel)
        result2 = generator2("Generate a JSON with name='test' and value=42", max_tokens=50)
        print(f"Result type: {type(result2)}")
        print(f"Result: {result2}")
        if hasattr(result2, 'model_dump'):
            print(f"Dumped: {result2.model_dump()}")
            
        # Test 3: Direct generation without Outlines
        print("\n3. Testing direct MLX generation (no Outlines):")
        from mlx_lm import generate as mlx_generate
        response = mlx_generate(
            mlx_model, 
            tokenizer,
            "Generate a simple JSON object: ",
            max_tokens=50,
            temp=0.1
        )
        print(f"Direct MLX response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def test_through_api():
    """Test through the API to see the actual response"""
    port_file = Path(".port")
    if port_file.exists():
        port = int(port_file.read_text().strip())
        base_url = f"http://localhost:{port}"
    else:
        base_url = "http://localhost:8000"
    
    print(f"\n\nTesting through API at {base_url}")
    print("=" * 60)
    
    import httpx
    
    async with httpx.AsyncClient() as client:
        # Get raw response to see exactly what's returned
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "medium",
                "messages": [{"role": "user", "content": "Generate JSON: {\"test\": 123}"}],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 50
            }
        )
        
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Raw response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"\nExtracted content: '{content}'")
            print(f"Content length: {len(content)}")
            print(f"First 20 chars repr: {repr(content[:20])}")


if __name__ == "__main__":
    print("Debugging MLX JSON generation...")
    asyncio.run(test_mlx_directly())
    asyncio.run(test_through_api())