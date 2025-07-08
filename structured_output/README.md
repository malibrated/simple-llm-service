# Structured Output Module

This module provides structured output generation for the LLM service, ensuring responses conform to specified formats using constrained generation.

## Current Status

### Supported Format

Currently, only `json_object` format is supported across all backends (llama.cpp and MLX) for consistent behavior:

```json
{
  "response_format": {
    "type": "json_object"
  }
}
```

This ensures the model generates valid JSON output without any markdown formatting or additional text.

### Architecture

The module implements an abstraction layer that:
1. Accepts requests in a standard format
2. Uses the requested model (no model switching)
3. Applies backend-specific constrained generation
4. Returns clean, structured responses

### Components

- **UnifiedStructuredGenerator**: Main entry point that coordinates backends
- **StructuredBackend**: Abstract interface for backend implementations
- **LlamaCppBackend**: Uses GBNF grammars for constrained generation
- **MLXBackend**: Uses Outlines library for constrained JSON generation
- **SchemaProcessor**: Validates and processes response formats

### Backend Implementations

#### LlamaCpp Backend
- Uses native GBNF (GGML BNF) grammar support
- Converts JSON schemas to GBNF grammars
- Provides hard constraints during generation
- Guarantees valid JSON output

#### MLX Backend
- Uses [Outlines](https://github.com/outlines-dev/outlines) library for constrained generation
- Leverages Pydantic models for schema definition
- Applies token-level constraints during generation
- Produces clean JSON without markdown wrapping

### Usage Example

```python
# In server.py
response_format = {
    "type": "json_object"
}

result = await model_manager.generate_structured(
    prompt=prompt,
    response_format=response_format,
    model_tier=model_tier,
    temperature=0.1
)

# Result contains clean JSON:
# {
#   "content": '{"name": "Product", "price": 29.99}',
#   "parsed_result": {"name": "Product", "price": 29.99},
#   "processing_time_ms": 150.5,
#   "backend_used": "mlx-outlines"  # or "llamacpp"
# }
```

### Requirements

For MLX constrained generation:
```bash
pip install outlines mlx-lm
```

### Future Enhancements

1. **Expand format support**:
   - JSON Schema validation (full OpenAI-compatible)
   - Custom GBNF grammars
   - Regex patterns
   - Function calling schemas

2. **Advanced features**:
   - Schema caching and optimization
   - Streaming structured output
   - Partial schema validation
   - Better error handling and recovery

3. **Performance optimizations**:
   - Pre-compiled schema caching
   - Batch generation support
   - Token probability analysis

### Testing

Run tests with:
```bash
# Test all backends
python test_structured_output.py

# Test MLX Outlines integration specifically
python test_mlx_outlines_correct.py
```

The tests verify:
- Clean JSON generation without markdown
- Consistent output across backends
- Proper error handling for unsupported formats
- Schema compliance