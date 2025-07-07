#!/usr/bin/env python3
"""
Simple test script to verify the LLM Service API is working.
"""
import requests
import json
import sys
from typing import Dict, Any


def test_endpoint(name: str, method: str, url: str, data: Dict[str, Any] = None) -> bool:
    """Test a single endpoint."""
    print(f"\nTesting {name}...")
    print("-" * 40)
    
    try:
        if method == "GET":
            response = requests.get(url)
        else:
            response = requests.post(url, json=data)
            
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:", json.dumps(result, indent=2)[:500])
            if len(json.dumps(result)) > 500:
                print("... (truncated)")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to service")
        print("Make sure the service is running: ./start_service.sh")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all API tests."""
    base_url = "http://localhost:8000"
    
    print("LLM Service API Test")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Health check
    tests_total += 1
    if test_endpoint("Health Check", "GET", f"{base_url}/health"):
        tests_passed += 1
    
    # Test 2: List models
    tests_total += 1
    if test_endpoint("List Models", "GET", f"{base_url}/v1/models"):
        tests_passed += 1
    
    # Test 3: Chat completion
    tests_total += 1
    chat_data = {
        "model": "medium",
        "messages": [
            {"role": "user", "content": "Say hello in one word"}
        ],
        "max_tokens": 10,
        "temperature": 0.1
    }
    if test_endpoint("Chat Completion", "POST", f"{base_url}/v1/chat/completions", chat_data):
        tests_passed += 1
    
    # Test 4: Text completion
    tests_total += 1
    completion_data = {
        "model": "light",
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0.1
    }
    if test_endpoint("Text Completion", "POST", f"{base_url}/v1/completions", completion_data):
        tests_passed += 1
    
    # Test 5: Structured output
    tests_total += 1
    structured_data = {
        "model": "medium",
        "messages": [
            {"role": "user", "content": 'Return JSON with a "status" field set to "ok"'}
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 20,
        "temperature": 0.1
    }
    if test_endpoint("Structured Output", "POST", f"{base_url}/v1/chat/completions", structured_data):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())