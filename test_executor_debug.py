#!/usr/bin/env python3
"""Debug executor issue"""
import asyncio

def test_func(prompt, **kwargs):
    print(f"test_func called with prompt='{prompt}' and kwargs={kwargs}")
    return "result"

async def main():
    loop = asyncio.get_event_loop()
    
    # Test 1: Direct call works
    print("Direct call:")
    result = test_func("hello", max_tokens=50)
    print(f"Result: {result}\n")
    
    # Test 2: Executor with kwargs
    print("Executor with wrapper:")
    def wrapper():
        return test_func("hello", max_tokens=50)
    
    result = await loop.run_in_executor(None, wrapper)
    print(f"Result: {result}\n")
    
    # Test 3: What's actually happening
    print("What might be happening:")
    try:
        # This would fail if the executor is passing kwargs incorrectly
        result = await loop.run_in_executor(None, wrapper, max_tokens=50)
    except TypeError as e:
        print(f"Error: {e}")

asyncio.run(main())