#!/usr/bin/env python3
"""
Examples of using structured output with the LLM service.
"""
import asyncio
import json
import httpx
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from pathlib import Path


# Example 1: Entity Extraction with JSON Schema
async def entity_extraction_example(base_url="http://localhost:8000"):
    """Extract named entities from text using JSON Schema."""
    print("\n=== Entity Extraction Example ===")
    
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "entity_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "The entity text"},
                                "type": {"type": "string", "enum": ["PERSON", "ORG", "LOCATION", "DATE", "MONEY"]},
                                "context": {"type": "string", "description": "Context around the entity"}
                            },
                            "required": ["text", "type"],
                            "additionalProperties": False
                        }
                    },
                    "summary": {"type": "string", "description": "Brief summary of the text"}
                },
                "required": ["entities"],
                "additionalProperties": False
            }
        }
    }
    
    text = """
    Microsoft announced that Satya Nadella will visit Tokyo next month to meet with 
    Japanese Prime Minister. The meeting is scheduled for March 15, 2024, and will 
    focus on AI investments worth $10 billion in the region.
    """
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "qwen3-8b",
                "messages": [
                    {"role": "system", "content": "Extract entities from the text."},
                    {"role": "user", "content": text}
                ],
                "response_format": schema,
                "temperature": 0.1
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            content = json.loads(result["choices"][0]["message"]["content"])
            print(f"Entities found: {len(content['entities'])}")
            for entity in content['entities']:
                print(f"  - {entity['text']} ({entity['type']})")


# Example 2: Structured Data Generation with GBNF
async def product_generation_example(base_url="http://localhost:8000"):
    """Generate structured product data using GBNF grammar."""
    print("\n=== Product Generation Example ===")
    
    # GBNF grammar for product data
    grammar = '''
root ::= product
product ::= "{" ws props ws "}"
props ::= prop_id ws "," ws prop_name ws "," ws prop_desc ws "," ws prop_price ws "," ws prop_stock
prop_id ::= "\\"id\\":" ws number
prop_name ::= "\\"name\\":" ws string
prop_desc ::= "\\"description\\":" ws string
prop_price ::= "\\"price\\":" ws price_value
prop_stock ::= "\\"in_stock\\":" ws boolean
price_value ::= number "." [0-9] [0-9]
string ::= "\\"" char* "\\""
char ::= [^"\\\\] | "\\\\" escape
escape ::= ["\\\\/bfnrt]
number ::= [1-9] [0-9]*
boolean ::= "true" | "false"
ws ::= [ \\t\\n\\r]*
'''
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "qwen3-8b",
                "messages": [
                    {"role": "system", "content": "Generate product data."},
                    {"role": "user", "content": "Create a laptop product with realistic details."}
                ],
                "response_format": {
                    "type": "gbnf_grammar",
                    "grammar": grammar
                },
                "temperature": 0.7
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            product = json.loads(content)
            print(f"Generated product:")
            print(f"  Name: {product['name']}")
            print(f"  Price: ${product['price']}")
            print(f"  In Stock: {product['in_stock']}")


# Example 3: Sentiment Analysis with Structured Output
async def sentiment_analysis_example(base_url="http://localhost:8000"):
    """Analyze sentiment with confidence scores."""
    print("\n=== Sentiment Analysis Example ===")
    
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "sentiment_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral", "mixed"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "aspects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "aspect": {"type": "string"},
                                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                                "keywords": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["aspect", "sentiment"]
                        }
                    },
                    "explanation": {"type": "string"}
                },
                "required": ["sentiment", "confidence", "explanation"]
            }
        }
    }
    
    reviews = [
        "The product quality is excellent, but the customer service was terrible. Shipping was fast though.",
        "Absolutely love this! Best purchase I've made all year. Highly recommend!",
        "It's okay, nothing special. Does what it's supposed to do."
    ]
    
    async with httpx.AsyncClient() as client:
        for review in reviews:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": "qwen3-8b",
                    "messages": [
                        {"role": "system", "content": "Analyze the sentiment of the review."},
                        {"role": "user", "content": review}
                    ],
                    "response_format": schema,
                    "temperature": 0.1
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = json.loads(result["choices"][0]["message"]["content"])
                print(f"\nReview: {review[:50]}...")
                print(f"Sentiment: {analysis['sentiment']} (confidence: {analysis['confidence']:.2f})")
                if 'aspects' in analysis:
                    print("Aspects:")
                    for aspect in analysis['aspects']:
                        print(f"  - {aspect['aspect']}: {aspect['sentiment']}")


# Example 4: Function Call Extraction
async def function_call_example(base_url="http://localhost:8000"):
    """Extract function calls from natural language."""
    print("\n=== Function Call Extraction Example ===")
    
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "function_call",
            "schema": {
                "type": "object",
                "properties": {
                    "function": {"type": "string", "enum": ["search_web", "send_email", "create_reminder", "get_weather"]},
                    "parameters": {
                        "type": "object",
                        "additionalProperties": True
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["function", "parameters"]
            }
        }
    }
    
    commands = [
        "Search the web for the latest Python tutorials",
        "Send an email to john@example.com about the meeting tomorrow",
        "What's the weather like in San Francisco?",
        "Remind me to call mom at 3 PM"
    ]
    
    async with httpx.AsyncClient() as client:
        for command in commands:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": "qwen3-8b",
                    "messages": [
                        {"role": "system", "content": "Extract the function call from the user's request."},
                        {"role": "user", "content": command}
                    ],
                    "response_format": schema,
                    "temperature": 0.1
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                call = json.loads(result["choices"][0]["message"]["content"])
                print(f"\nCommand: {command}")
                print(f"Function: {call['function']}")
                print(f"Parameters: {json.dumps(call['parameters'], indent=2)}")


# Example 5: Using Pydantic Models (converted to JSON Schema)
class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Task(BaseModel):
    title: str = Field(..., description="Task title")
    description: Optional[str] = Field(None, description="Detailed description")
    priority: TaskPriority = Field(TaskPriority.MEDIUM, description="Task priority")
    due_date: Optional[str] = Field(None, description="Due date in YYYY-MM-DD format")
    tags: List[str] = Field(default_factory=list, description="Task tags")
    estimated_hours: Optional[float] = Field(None, ge=0, description="Estimated hours to complete")


class TaskList(BaseModel):
    tasks: List[Task] = Field(..., description="List of tasks")
    total_estimated_hours: float = Field(..., description="Total estimated hours for all tasks")


async def pydantic_model_example(base_url="http://localhost:8000"):
    """Use Pydantic models for structured output."""
    print("\n=== Pydantic Model Example ===")
    
    # Convert Pydantic model to JSON Schema
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "task_list",
            "schema": TaskList.model_json_schema()
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "qwen3-8b",
                "messages": [
                    {"role": "system", "content": "Generate a task list."},
                    {"role": "user", "content": "Create a task list for building a web application MVP"}
                ],
                "response_format": schema,
                "temperature": 0.7
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            content = json.loads(result["choices"][0]["message"]["content"])
            task_list = TaskList(**content)  # Validate with Pydantic
            
            print(f"Generated {len(task_list.tasks)} tasks")
            print(f"Total estimated hours: {task_list.total_estimated_hours}")
            for i, task in enumerate(task_list.tasks, 1):
                print(f"\n{i}. {task.title} [{task.priority.value}]")
                if task.description:
                    print(f"   {task.description}")
                if task.due_date:
                    print(f"   Due: {task.due_date}")


async def main():
    """Run all examples."""
    print("Structured Output Examples")
    print("=" * 50)
    
    # Read port from .port file
    port_file = Path(".port")
    if port_file.exists():
        port = int(port_file.read_text().strip())
        base_url = f"http://localhost:{port}"
        print(f"Using LLM service on port {port}")
    else:
        base_url = "http://localhost:8000"
        print("No .port file found, using default port 8000")
    
    try:
        # Check service is running
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/health")
            if response.status_code != 200:
                print("Service is not running!")
                return
        
        # Run examples
        await entity_extraction_example(base_url)
        await product_generation_example(base_url)
        await sentiment_analysis_example(base_url)
        await function_call_example(base_url)
        await pydantic_model_example(base_url)
        
        print("\n" + "=" * 50)
        print("All examples completed!")
        
    except httpx.ConnectError:
        print(f"Cannot connect to the service. Make sure it's running on {base_url}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())