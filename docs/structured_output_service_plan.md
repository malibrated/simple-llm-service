# Structured Output Service Implementation Plan

## Executive Summary

This document outlines the implementation plan for adding structured output capabilities to the existing LLM service. The implementation will extend the current OpenAI-compatible API to support JSON Schema, GBNF grammars, and other structured output formats while maintaining backward compatibility.

## 1. Architecture Overview

### 1.1 Integration Approach
- **Extend existing service**: Integrate into current `server.py` and `model_manager.py`
- **Factored modules**: Create separate `structured_output/` module for grammar/schema logic
- **Unified interface**: Single API that abstracts backend differences (llama.cpp vs MLX)
- **Leverage existing infrastructure**: Use current model tiers and queue system

### 1.2 Key Design Principles
- **Stateless**: Schemas are transient, passed per request
- **Backend agnostic**: Clients remain unaware of underlying inference engine
- **OpenAI compatible**: Follow latest OpenAI structured outputs specification
- **Concurrent**: Support multiple simultaneous structured generation requests
- **Extensible**: Easy to add new schema types or backends

## 2. Current State Analysis

### 2.1 Existing Capabilities
- Basic grammar support via `grammar` parameter in requests
- Simple JSON grammar implementation (`_get_json_grammar()`)
- llama.cpp backend supports GBNF grammars via `LlamaGrammar`
- MLX backend exists but lacks grammar support
- Model manager with lazy loading and queue-based processing

### 2.2 Gaps to Address
- No JSON Schema to GBNF conversion
- No MLX/Outlines integration for structured output
- Limited schema validation
- No support for OpenAI's `json_schema` response format
- No refusal handling for structured outputs

## 3. Implementation Design

### 3.1 Module Structure
```
llmservice/
├── server.py                      # Extended with new response_format handling
├── model_manager.py               # Extended with structured generation methods
└── structured_output/             # New module
    ├── __init__.py
    ├── interfaces.py              # Abstract interfaces (StructuredBackend)
    ├── schema_processor.py        # Schema validation and type detection
    ├── backends/
    │   ├── __init__.py
    │   ├── llamacpp_backend.py    # GBNF grammar support
    │   └── mlx_backend.py         # Outlines-MLX support
    ├── converters/
    │   ├── __init__.py
    │   ├── json_to_gbnf.py        # JSON Schema → GBNF converter
    │   └── pydantic_to_gbnf.py   # Pydantic → GBNF converter
    └── unified_generator.py       # Backend-agnostic interface
```

### 3.2 API Extensions

#### Request Model Updates
```python
class ChatCompletionRequest(BaseModel):
    # ... existing fields ...
    response_format: Optional[Dict[str, Any]] = None  # Extended support
    
    # response_format examples:
    # {"type": "json_object"}  # Basic JSON mode (existing)
    # {
    #     "type": "json_schema",
    #     "json_schema": {
    #         "name": "entity_extraction",
    #         "strict": true,
    #         "schema": {...}  # JSON Schema definition
    #     }
    # }
    # {"type": "gbnf_grammar", "grammar": "..."}  # GBNF grammar
    # {"type": "regex", "pattern": "..."}         # Regex pattern
```

#### Response Updates
- Add `refusal` field to handle cases where model refuses to generate
- Ensure backward compatibility with existing response format

### 3.3 Core Components

#### 3.3.1 Schema Processor
```python
class SchemaProcessor:
    """Validates and processes different schema formats"""
    
    def validate_response_format(self, response_format: Dict) -> bool
    def extract_schema(self, response_format: Dict) -> Any
    def get_schema_type(self, response_format: Dict) -> SchemaType
```

#### 3.3.2 Structured Backend Interface
```python
class StructuredBackend(ABC):
    """Abstract interface for structured generation backends"""
    
    @abstractmethod
    async def generate(self, request: StructuredRequest) -> StructuredResponse
    
    @abstractmethod
    def supports_format_type(self, format_type: str) -> bool
    
    @abstractmethod
    def compile_schema(self, response_format: Dict) -> Any
```

#### 3.3.3 JSON to GBNF Converter
```python
class JSONSchemaToGBNF:
    """Converts JSON Schema to GBNF grammar for llama.cpp"""
    
    def convert(self, json_schema: Dict) -> str
    def handle_object(self, schema: Dict) -> str
    def handle_array(self, schema: Dict) -> str
    def handle_primitives(self, schema: Dict) -> str
```

#### 3.3.4 Unified Generator
```python
class UnifiedStructuredGenerator:
    """Backend-agnostic structured generation coordinator"""
    
    def __init__(self, model_manager: ModelManager)
    async def generate(self, prompt: str, response_format: Dict, **kwargs)
    def select_backend(self, format_type: str, model_config: Dict) -> StructuredBackend
```

### 3.4 Integration Points

#### 3.4.1 Model Manager Extensions
```python
class ModelManager:
    # ... existing code ...
    
    async def generate_structured(self, 
                                prompt: str,
                                response_format: Dict[str, Any],
                                model_tier: ModelTier,
                                **kwargs) -> Dict[str, Any]:
        """Generate text with structured output constraints"""
        
    async def generate_with_grammar(self, 
                                  prompt: str, 
                                  grammar: str,
                                  model_tier: ModelTier,
                                  **kwargs) -> Dict[str, Any]:
        """Generate text with GBNF grammar (for llama.cpp backend)"""
```

#### 3.4.2 Server.py Updates
```python
# In chat_completion method:
if request.response_format:
    format_type = request.response_format.get("type")
    
    if format_type == "json_schema":
        # Use structured generation with JSON Schema
        result = await self.model_manager.generate_structured(...)
    elif format_type == "gbnf_grammar":
        # Use direct GBNF grammar
        generation_params["grammar"] = request.response_format["grammar"]
    # ... handle other types
```

## 4. Implementation Phases

### Phase 1: Core Foundation (Days 1-3)
1. **Day 1**: Create structured_output module structure
   - Set up directory structure
   - Create base interfaces and data models
   - Implement SchemaProcessor for validation

2. **Day 2**: Build JSON Schema to GBNF converter
   - Handle basic types (object, array, string, number, boolean)
   - Support required fields and basic constraints
   - Add comprehensive test cases

3. **Day 3**: Extend ModelManager
   - Add generate_structured method
   - Enhance generate_with_grammar for better grammar support
   - Ensure backward compatibility

### Phase 2: Backend Implementation (Days 4-6)
4. **Day 4**: Implement LlamaCppBackend
   - Integrate with existing LlamaCppWrapper
   - Add grammar compilation and caching
   - Handle different schema types

5. **Day 5**: Implement MLXBackend (if outlines available)
   - Install and test outlines-core with MLX
   - Create MLX structured generation wrapper
   - Handle MLX-specific constraints

6. **Day 6**: Create UnifiedGenerator
   - Implement backend selection logic
   - Add request routing
   - Test with both backends

### Phase 3: API Integration (Days 7-8)
7. **Day 7**: Update API endpoints
   - Extend ChatCompletionRequest model
   - Update chat_completions handler
   - Add proper error handling

8. **Day 8**: Advanced features
   - Add schema compilation caching
   - Implement refusal detection
   - Add validation for generated outputs

### Phase 4: Testing & Documentation (Days 9-10)
9. **Day 9**: Comprehensive testing
   - Unit tests for converters
   - Integration tests for backends
   - End-to-end API tests
   - Performance benchmarking

10. **Day 10**: Documentation and examples
    - Update API documentation
    - Create usage examples
    - Document supported schema features
    - Performance tuning guide

## 5. Technical Specifications

### 5.1 Supported Schema Types
1. **JSON Schema** (OpenAI compatible)
   - Basic types: object, array, string, number, boolean, null
   - Constraints: required, minLength, maxLength, minimum, maximum
   - Patterns: enum, const
   - Complex: oneOf, anyOf (limited support)

2. **GBNF Grammar** (llama.cpp native)
   - Direct grammar specification
   - Full GBNF syntax support

3. **Regex Patterns** (simple constraints)
   - For specific format requirements

4. **Pydantic Models** (future)
   - Convert to JSON Schema internally

### 5.2 Backend Capabilities Matrix

| Feature | llama.cpp | MLX/Outlines |
|---------|-----------|--------------|
| JSON Schema | ✓ (via GBNF) | ✓ (native) |
| GBNF Grammar | ✓ (native) | ✗ |
| Regex | ✓ (via GBNF) | ✓ (native) |
| Performance | Fast | Fast |
| GPU Support | ✓ | ✓ (Apple Silicon) |

### 5.3 Error Handling

1. **Schema Validation Errors**
   - Return 400 with clear error message
   - Specify which part of schema is invalid

2. **Generation Failures**
   - Retry with relaxed constraints
   - Fall back to unstructured generation
   - Return partial results if possible

3. **Backend Errors**
   - Automatic fallback to alternative backend
   - Clear error messages in logs

## 6. Example Usage

### 6.1 Basic JSON Schema
```python
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Extract entities from: Apple Inc was founded by Steve Jobs"}],
    "response_format": {
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
                                "text": {"type": "string"},
                                "type": {"type": "string", "enum": ["PERSON", "ORG", "LOCATION"]}
                            },
                            "required": ["text", "type"]
                        }
                    }
                },
                "required": ["entities"]
            }
        }
    }
})
```

### 6.2 GBNF Grammar
```python
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Generate a person's details"}],
    "response_format": {
        "type": "gbnf_grammar",
        "grammar": '''
root ::= person
person ::= "{"  "\\\"name\\\":" ws string "," ws "\\\"age\\\":" ws number "}"
string ::= "\\"" [^"]* "\\""
number ::= [0-9]+
ws ::= [ \\t\\n]*
        '''
    }
})
```

## 7. Success Criteria

1. **Functional Requirements**
   - ✓ Support OpenAI-compatible json_schema format
   - ✓ Support custom GBNF grammars
   - ✓ Work with both llama.cpp and MLX backends
   - ✓ Handle concurrent requests
   - ✓ Validate schemas and outputs

2. **Performance Requirements**
   - Schema compilation < 100ms
   - No significant latency increase vs unstructured
   - Support 100+ concurrent requests
   - Efficient memory usage for compiled schemas

3. **Quality Requirements**
   - 95%+ success rate for valid schemas
   - Clear error messages for invalid inputs
   - Comprehensive test coverage (>80%)
   - Well-documented API

## 8. Future Enhancements

1. **Advanced Schema Support**
   - Complex JSON Schema features (allOf, dependencies)
   - XML Schema support
   - Protocol Buffer schemas

2. **Performance Optimizations**
   - Persistent schema compilation cache
   - Pre-compiled common schemas
   - Grammar optimization algorithms

3. **Developer Experience**
   - Schema builder UI
   - Validation playground
   - Performance profiling tools

4. **Additional Backends**
   - vLLM with XGrammar
   - TensorRT-LLM
   - Custom inference servers

## 9. Risk Mitigation

1. **Technical Risks**
   - MLX/Outlines compatibility → Fallback to llama.cpp only
   - Complex schema support → Start with basic features
   - Performance degradation → Extensive benchmarking

2. **Implementation Risks**
   - Scope creep → Strict phase boundaries
   - Backend differences → Abstract interface design
   - Breaking changes → Comprehensive testing

## 10. Conclusion

This implementation plan provides a clear path to add structured output capabilities while:
- Maintaining backward compatibility
- Leveraging existing infrastructure
- Following OpenAI API standards
- Supporting multiple backends
- Enabling future enhancements

The phased approach ensures quick wins while building toward a comprehensive solution.