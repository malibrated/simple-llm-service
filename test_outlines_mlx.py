#!/usr/bin/env python3
"""Test Outlines MLX integration directly"""
import sys
import os
sys.path.insert(0, '.')

from mlx_lm import load
from outlines.models import from_mlxlm
from outlines import generate
from pydantic import BaseModel

# Load env
from dotenv import load_dotenv
load_dotenv()

model_path = os.getenv("MEDIUM_MODEL_PATH")
print(f"Loading model from: {model_path}")

# Load model
mlx_model, tokenizer = load(model_path)
model = from_mlxlm(mlx_model, tokenizer)

# Create flexible model
class FlexibleJSON(BaseModel):
    class Config:
        extra = 'allow'

# Create generator
json_generator = generate.json(model, FlexibleJSON)

# Test 1: Direct call
print("\n1. Testing direct generator call:")
try:
    result = json_generator("Generate a product with name and price")
    print(f"Result: {result}")
    print(f"Type: {type(result)}")
    if hasattr(result, 'model_dump'):
        print(f"Dumped: {result.model_dump()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: With parameters
print("\n2. Testing with parameters:")
try:
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.1)
    result = json_generator("Generate a product with name and price", sampler=sampler, max_tokens=50)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    
# Test 3: Check signature
print("\n3. Checking generator signature:")
import inspect
print(f"Signature: {inspect.signature(json_generator)}")

# Test 4: Check what the generator actually is
print(f"\n4. Generator type: {type(json_generator)}")
print(f"Generator attrs: {dir(json_generator)}")