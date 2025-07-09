#!/usr/bin/env python3
"""
Test script to verify json_object format returns clean JSON without markdown.
"""
import asyncio
import json
import httpx
from pathlib import Path
import sys


async def test_json_object_format():
    """Test that response_format json_object returns clean JSON."""
    # Read port from .port file
    port_file = Path(".port")
    if port_file.exists():
        port = int(port_file.read_text().strip())
        base_url = f"http://localhost:{port}"
    else:
        base_url = "http://localhost:8000"
    
    print(f"Testing JSON object format at {base_url}")
    print("=" * 60)
    
    # Test cases with different prompts that might trigger markdown
    test_cases = [
        {
            "name": "Basic product generation",
            "prompt": "Generate a product listing with name, price, and description"
        },
        {
            "name": "Entity extraction", 
            "prompt": "Extract entities from this text as JSON: Apple Inc. announced that Tim Cook will visit Tokyo next month. Return people, organizations, and locations."
        },
        {
            "name": "Complex nested structure",
            "prompt": "Create a user profile with nested objects. Include: name, age, address (with street, city, zip), hobbies (array), and preferences (object with theme, language)"
        },
        {
            "name": "List generation",
            "prompt": "Generate a JSON array of 3 programming languages with their key features"
        }
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for test_case in test_cases:
            print(f"\nTest: {test_case['name']}")
            print("-" * 40)
            
            request = {
                "model": "light",  # Try with light model first
                "messages": [
                    {"role": "user", "content": test_case['prompt']}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 300
            }
            
            try:
                response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    json=request
                )
                
                if response.status_code != 200:
                    print(f"❌ Error {response.status_code}: {response.text}")
                    continue
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Check for markdown indicators
                has_markdown = False
                markdown_indicators = ["```json", "```", "**", "##", "###"]
                for indicator in markdown_indicators:
                    if indicator in content:
                        has_markdown = True
                        print(f"❌ Found markdown indicator: {indicator}")
                        break
                
                # Try to parse as JSON
                try:
                    parsed = json.loads(content)
                    if not has_markdown:
                        print(f"✅ Clean JSON returned")
                        print(f"   Content preview: {json.dumps(parsed, indent=2)[:200]}...")
                    else:
                        print(f"⚠️  JSON parseable but contains markdown")
                        print(f"   Raw content: {content[:200]}...")
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON: {e}")
                    print(f"   Raw content: {content[:200]}...")
                    
            except Exception as e:
                print(f"❌ Request failed: {e}")
        
        # Also test with medium model if available
        print("\n\nTesting with medium model...")
        print("=" * 60)
        
        request = {
            "model": "medium",
            "messages": [
                {"role": "user", "content": "Generate a simple JSON object with name and value fields"}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 100
        }
        
        try:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json=request
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                print(f"✅ Medium model clean JSON: {json.dumps(parsed, indent=2)}")
            else:
                print(f"❌ Medium model error: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️  Medium model not available or error: {e}")


async def test_without_response_format():
    """Test what happens without response_format for comparison."""
    # Read port from .port file
    port_file = Path(".port")
    if port_file.exists():
        port = int(port_file.read_text().strip())
        base_url = f"http://localhost:{port}"
    else:
        base_url = "http://localhost:8000"
    
    print("\n\nTesting WITHOUT response_format (for comparison)")
    print("=" * 60)
    
    request = {
        "model": "light",
        "messages": [
            {"role": "user", "content": "Generate a JSON object with name and price fields"}
        ],
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json=request
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"Without response_format: {content[:200]}...")
            
            # Check if it naturally includes markdown
            if "```" in content:
                print("⚠️  Model naturally includes markdown without constraints")
            else:
                print("✅ Model returned clean text without markdown")


if __name__ == "__main__":
    print("Testing JSON object format fix...")
    asyncio.run(test_json_object_format())
    asyncio.run(test_without_response_format())
    print("\n✅ Test complete!")