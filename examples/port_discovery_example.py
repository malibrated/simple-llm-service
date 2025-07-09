#!/usr/bin/env python3
"""
Example of discovering the LLM service port dynamically.
"""
import os
from pathlib import Path
import requests
import time


def discover_llm_service_port(port_file=".port", default_port=8000):
    """
    Discover the port where LLM service is running.
    
    Args:
        port_file: Path to the port file (default: .port)
        default_port: Default port if file doesn't exist
        
    Returns:
        int: The port number
    """
    port_path = Path(port_file)
    
    if port_path.exists():
        try:
            port = int(port_path.read_text().strip())
            print(f"Discovered LLM service on port {port} from {port_file}")
            return port
        except Exception as e:
            print(f"Error reading port file: {e}")
    
    print(f"Using default port {default_port}")
    return default_port


def check_service_health(port):
    """Check if the service is healthy."""
    try:
        response = requests.get(f"http://localhost:{port}/health")
        if response.status_code == 200:
            print(f"✓ Service is healthy on port {port}")
            return True
    except requests.exceptions.ConnectionError:
        print(f"✗ Service not responding on port {port}")
    return False


def main():
    # Discover the port
    port = discover_llm_service_port()
    
    # Check if service is running
    if not check_service_health(port):
        print("\nService not running. Please start it with ./start_service.sh")
        return
    
    # Use the discovered port to make API calls
    base_url = f"http://localhost:{port}"
    
    # List available models
    response = requests.get(f"{base_url}/v1/models")
    if response.status_code == 200:
        models = response.json()
        print(f"\nAvailable models:")
        for model in models["data"]:
            print(f"  - {model['id']}")
    
    # Make a chat completion request
    chat_request = {
        "model": "light",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.1
    }
    
    print(f"\nMaking chat request to {base_url}/v1/chat/completions")
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=chat_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['choices'][0]['message']['content']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    main()