#!/usr/bin/env python3
"""
Test script for embedding functionality.
"""
import requests
import json
import numpy as np
from pathlib import Path


def test_embeddings():
    """Test the embedding endpoint."""
    # Check if service is running
    port_file = Path(".port")
    if port_file.exists():
        port = int(port_file.read_text().strip())
    else:
        port = 8000
    
    base_url = f"http://localhost:{port}"
    
    # Check health
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            print(f"Service not healthy: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"Service not running on port {port}")
        print("Please start the service with ./start_service.sh")
        return
    
    print(f"Testing embeddings on port {port}...")
    
    # Test single text embedding
    print("\n1. Testing single text embedding:")
    single_request = {
        "model": "light",
        "input": "The quick brown fox jumps over the lazy dog"
    }
    
    response = requests.post(
        f"{base_url}/v1/embeddings",
        json=single_request
    )
    
    if response.status_code == 200:
        result = response.json()
        embedding = result["data"][0]["embedding"]
        print(f"✓ Single embedding generated")
        print(f"  - Dimension: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
        print(f"  - Tokens used: {result['usage']['total_tokens']}")
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
        return
    
    # Test batch embeddings
    print("\n2. Testing batch embeddings:")
    batch_request = {
        "model": "light",
        "input": [
            "First text to embed",
            "Second text to embed",
            "Third text to embed"
        ]
    }
    
    response = requests.post(
        f"{base_url}/v1/embeddings",
        json=batch_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Batch embeddings generated")
        print(f"  - Number of embeddings: {len(result['data'])}")
        for i, data in enumerate(result["data"]):
            print(f"  - Text {i+1} dimension: {len(data['embedding'])}")
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
        return
    
    # Test dimension reduction
    print("\n3. Testing dimension reduction:")
    dim_request = {
        "model": "light",
        "input": "Test embedding with dimension reduction",
        "dimensions": 512  # Request smaller dimension
    }
    
    response = requests.post(
        f"{base_url}/v1/embeddings",
        json=dim_request
    )
    
    if response.status_code == 200:
        result = response.json()
        embedding = result["data"][0]["embedding"]
        print(f"✓ Dimension reduction worked")
        print(f"  - Requested dimension: 512")
        print(f"  - Actual dimension: {len(embedding)}")
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
    
    # Test cosine similarity
    print("\n4. Testing semantic similarity:")
    texts = [
        "The weather is beautiful today",
        "It's a lovely sunny day",
        "Python is a programming language"
    ]
    
    response = requests.post(
        f"{base_url}/v1/embeddings",
        json={"model": "light", "input": texts}
    )
    
    if response.status_code == 200:
        result = response.json()
        embeddings = [np.array(d["embedding"]) for d in result["data"]]
        
        # Calculate cosine similarities
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        print("✓ Cosine similarities:")
        print(f"  - '{texts[0]}' vs '{texts[1]}': {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
        print(f"  - '{texts[0]}' vs '{texts[2]}': {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
        print(f"  - '{texts[1]}' vs '{texts[2]}': {cosine_similarity(embeddings[1], embeddings[2]):.4f}")
        print("\n  (Higher values = more similar, expecting first two to be most similar)")
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")


if __name__ == "__main__":
    test_embeddings()