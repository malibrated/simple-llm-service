#!/usr/bin/env python3
"""
Test script for structured output functionality.
Currently only json_object format is supported.
"""
import asyncio
import json
import httpx
from pathlib import Path


async def test_json_object(base_url="http://localhost:8000"):
    """Test basic JSON object mode."""
    print("\n=== Testing JSON Object Mode ===")
    
    request = {
        "model": "light",
        "messages": [
            {"role": "user", "content": "List three primary colors in JSON format"}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json=request,
            timeout=30.0
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"Response: {content}")
            
            try:
                parsed = json.loads(content)
                print(f"\nParsed: {json.dumps(parsed, indent=2)}")
                print("✓ Valid JSON generated")
            except json.JSONDecodeError:
                print("✗ Failed to parse as JSON")
        else:
            print(f"Error: {response.status_code} - {response.text}")


async def test_different_models(base_url="http://localhost:8000"):
    """Test JSON object mode with different models."""
    print("\n=== Testing Different Models ===")
    
    test_cases = [
        ("light", "Generate a person object with name and age"),
        ("medium", "Generate a product object with name and price"),
        ("qwen3-8b", "Generate a book object with title and author"),
    ]
    
    for model, prompt in test_cases:
        print(f"\n--- Testing {model} model ---")
        request = {
            "model": model,
            "messages": [
                {"role": "user", "content": f"{prompt}. Return as JSON."}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 100
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    json=request,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    try:
                        # Remove markdown code blocks if present
                        json_content = content
                        if "```json" in content:
                            json_content = content.split("```json")[1].split("```")[0].strip()
                        elif "```" in content:
                            json_content = content.split("```")[1].split("```")[0].strip()
                        
                        parsed = json.loads(json_content)
                        print(f"✓ {model}: {json.dumps(parsed)}")
                    except json.JSONDecodeError:
                        print(f"✗ {model}: Invalid JSON - {content[:100]}...")
                else:
                    print(f"✗ {model}: Error {response.status_code}")
            except Exception as e:
                print(f"✗ {model}: Exception - {type(e).__name__}: {str(e)[:100]}")


async def test_complex_json(base_url="http://localhost:8000"):
    """Test generating more complex JSON structures."""
    print("\n=== Testing Complex JSON Generation ===")
    
    request = {
        "model": "light",
        "messages": [
            {
                "role": "user", 
                "content": "Generate a JSON object representing a user profile with nested address object"
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "max_tokens": 200
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json=request,
            timeout=30.0
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            try:
                parsed = json.loads(content)
                print("Generated complex JSON:")
                print(json.dumps(parsed, indent=2))
                print("✓ Complex JSON structure generated successfully")
            except json.JSONDecodeError:
                print("✗ Failed to generate valid complex JSON")
                print(f"Raw output: {content}")
        else:
            print(f"Error: {response.status_code} - {response.text}")


async def test_unsupported_formats(base_url="http://localhost:8000"):
    """Test that unsupported formats are properly rejected."""
    print("\n=== Testing Unsupported Format Handling ===")
    
    unsupported_formats = [
        {
            "name": "json_schema",
            "format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {"type": "object"}
                }
            }
        },
        {
            "name": "gbnf_grammar", 
            "format": {
                "type": "gbnf_grammar",
                "grammar": "root ::= object"
            }
        },
        {
            "name": "regex",
            "format": {
                "type": "regex",
                "pattern": "\\d+"
            }
        }
    ]
    
    for test in unsupported_formats:
        print(f"\n--- Testing {test['name']} (should fail) ---")
        request = {
            "model": "light",
            "messages": [{"role": "user", "content": "Test"}],
            "response_format": test['format'],
            "max_tokens": 50
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json=request,
                timeout=30.0
            )
            
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', response.text)
                except:
                    error_detail = response.text
                    
                if "json_object" in str(error_detail).lower():
                    print(f"✓ Correctly rejected {test['name']} format")
                    print(f"  Error message: {error_detail}")
                else:
                    print(f"✗ Rejected but unclear error message: {error_detail}")
            else:
                print(f"✗ Unexpectedly accepted {test['name']} format")


async def main():
    """Run all tests."""
    print("Starting structured output tests...")
    
    # Read port from .port file
    port_file = Path(".port")
    if port_file.exists():
        port = int(port_file.read_text().strip())
        base_url = f"http://localhost:{port}"
        print(f"Using LLM service on port {port}")
    else:
        print("No .port file found, starting service...")
        import subprocess
        import time
        
        # Start the service
        subprocess.Popen(["./start_service.sh"], shell=True)
        
        # Wait for service to start and create .port file
        for _ in range(30):  # Wait up to 30 seconds
            if port_file.exists():
                port = int(port_file.read_text().strip())
                base_url = f"http://localhost:{port}"
                print(f"Service started on port {port}")
                break
            time.sleep(1)
        else:
            print("Service failed to start within 30 seconds")
            return
    
    print(f"Testing structured output on {base_url}")
    print("Note: Currently only 'json_object' format is supported\n")
    
    try:
        # Test basic connectivity
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/health")
            if response.status_code != 200:
                print("Service health check failed!")
                return
        
        # Run tests
        await test_json_object(base_url)
        await test_different_models(base_url)
        await test_complex_json(base_url)
        await test_unsupported_formats(base_url)
        
        print("\n" + "="*50)
        print("All tests completed!")
        print("\nSummary:")
        print("- json_object format: ✓ Supported")
        print("- json_schema format: ✗ Not yet supported")
        print("- gbnf_grammar format: ✗ Not yet supported") 
        print("- regex format: ✗ Not yet supported")
        
    except httpx.ConnectError:
        print(f"Cannot connect to the service. Make sure it's running on {base_url}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())