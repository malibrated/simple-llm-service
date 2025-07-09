#!/usr/bin/env python3
"""Minimal test to isolate MLX issue"""
import os
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

# Test 1: Direct MLX generation
print("1. Testing direct MLX generation:")
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model_path = os.getenv("MEDIUM_MODEL_PATH")
model, tokenizer = load(model_path)

# Create sampler
sampler = make_sampler(temp=0.1)

# Test direct generation
prompt = "Generate JSON: {\"test\": true}"
result = generate(model, tokenizer, prompt, max_tokens=50, sampler=sampler, verbose=False)
print(f"Result: {result}")

# Test 2: With Outlines wrapper
print("\n2. Testing with Outlines wrapper:")
from outlines.models import from_mlxlm

wrapped_model = from_mlxlm(model, tokenizer)
result2 = wrapped_model.generate(prompt, max_tokens=50)
print(f"Result: {result2}")

# Test 3: Check if the issue is with how Outlines passes kwargs
print("\n3. Testing Outlines with different parameters:")
try:
    # This might fail if Outlines doesn't handle kwargs properly
    result3 = wrapped_model.generate(prompt, max_tokens=50, temperature=0.1)
    print(f"Result: {result3}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()