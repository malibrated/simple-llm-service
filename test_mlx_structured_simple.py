#!/usr/bin/env python3
"""Simple test of MLX structured output"""
import httpx
import asyncio
import json
from pathlib import Path

async def test():
    # Get port
    port_file = Path(".port")
    if port_file.exists():
        port = int(port_file.read_text().strip())
        base_url = f"http://localhost:{port}"
    else:
        base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient(timeout=30) as client:
        # Test 1: Basic generation works
        print("1. Testing basic MLX generation:")
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "medium",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 10
            }
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            print(f"Response: {content}")
        
        # Test 2: With response_format
        print("\n2. Testing with response_format:")
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "medium",
                "messages": [{"role": "user", "content": "Return: {\"status\": \"ok\"}"}],
                "response_format": {"type": "json_object"},
                "max_tokens": 50
            }
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Full response: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    asyncio.run(test())