#!/usr/bin/env python3
"""
Example of using LLM Service with Langchain.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


def test_basic_chat():
    """Test basic chat functionality."""
    print("Testing basic chat...")
    
    # Create LLM instance pointing to local service
    llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # API key not required for local service
        model="medium",
        temperature=0.3,
        max_tokens=200
    )
    
    # Test direct invocation
    response = llm.invoke("What is the capital of France?")
    print(f"Response: {response.content}\n")
    
    # Test with messages
    messages = [
        SystemMessage(content="You are a helpful geography teacher."),
        HumanMessage(content="Name three major rivers in Europe.")
    ]
    response = llm.invoke(messages)
    print(f"Rivers response: {response.content}\n")


def test_chain():
    """Test Langchain chain functionality."""
    print("Testing chain...")
    
    # Create LLM
    llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
        model="medium",
        temperature=0.5
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "{text}")
    ])
    
    # Create chain
    chain = prompt | llm
    
    # Test chain
    result = chain.invoke({
        "input_language": "English",
        "output_language": "French",
        "text": "Hello, how are you today?"
    })
    print(f"Translation: {result.content}\n")


def test_structured_output():
    """Test structured output with JSON."""
    print("Testing structured output...")
    
    # Create LLM with JSON response format
    llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
        model="medium",
        temperature=0.1,
        model_kwargs={
            "response_format": {"type": "json_object"}
        }
    )
    
    # Create prompt for extraction
    prompt = """Extract the person's information from this text and return as JSON:
    
    John Smith is 32 years old and works as a software engineer in San Francisco.
    
    Return JSON with keys: name, age, occupation, location"""
    
    response = llm.invoke(prompt)
    print(f"Extracted data: {response.content}\n")


def test_streaming():
    """Test streaming functionality."""
    print("Testing streaming...")
    
    # Create LLM with streaming
    llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
        model="medium",
        streaming=True,
        temperature=0.7
    )
    
    # Stream response
    print("Streaming response: ", end="", flush=True)
    for chunk in llm.stream("Tell me a very short story about a robot."):
        print(chunk.content, end="", flush=True)
    print("\n")


def test_model_tiers():
    """Test different model tiers."""
    print("Testing model tiers...")
    
    tiers = ["light", "medium", "heavy"]
    prompt = "Explain quantum computing in one sentence."
    
    for tier in tiers:
        try:
            llm = ChatOpenAI(
                base_url="http://localhost:8000/v1",
                api_key="not-needed",
                model=tier,
                temperature=0.3,
                max_tokens=100
            )
            
            response = llm.invoke(prompt)
            print(f"{tier.upper()} tier: {response.content}")
        except Exception as e:
            print(f"{tier.upper()} tier: Not available - {e}")
    print()


def main():
    """Run all tests."""
    print("LLM Service Langchain Example")
    print("=" * 50)
    print("Make sure the LLM service is running on http://localhost:8000")
    print("=" * 50)
    print()
    
    try:
        test_basic_chat()
        test_chain()
        test_structured_output()
        test_model_tiers()
        # test_streaming()  # Uncomment if streaming is implemented
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the LLM service is running:")
        print("  cd /Users/patrickpark/Documents/Work/utils/llmservice")
        print("  python server.py")


if __name__ == "__main__":
    main()