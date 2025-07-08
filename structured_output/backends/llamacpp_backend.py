"""
Llama.cpp backend for structured output generation.
"""
import asyncio
import json
import time
from typing import Dict, Any, Optional

from ..interfaces import StructuredBackend, StructuredRequest, StructuredResponse
from ..schema_processor import SchemaType
from ..converters import JSONSchemaToGBNF


class LlamaCppBackend(StructuredBackend):
    """Llama.cpp backend with GBNF grammar support."""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.json_to_gbnf = JSONSchemaToGBNF()
        self._compiled_grammars: Dict[str, str] = {}  # Cache compiled grammars
    
    async def generate(self, request: StructuredRequest) -> StructuredResponse:
        """Generate structured output using llama.cpp with grammar."""
        start_time = time.perf_counter()
        
        # Get or compile grammar
        grammar = self.compile_schema(request.response_format)
        
        # Add grammar to generation parameters
        generation_params = request.generation_params.copy()
        generation_params['grammar'] = grammar
        
        # Get model tier from config
        model_tier = request.model_config.get('tier')
        
        # Generate using model manager
        result = await self.model_manager.generate(
            model_tier=model_tier,
            prompt=request.prompt,
            **generation_params
        )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Parse result based on format type
        parsed_result = self._parse_result(result['text'], request.response_format)
        
        return StructuredResponse(
            content=result['text'],
            parsed_result=parsed_result,
            processing_time_ms=processing_time,
            backend_used="llamacpp"
        )
    
    def supports_format_type(self, format_type: str) -> bool:
        """Check if backend supports given format type."""
        # For now, only support json_object for consistent behavior across backends
        return format_type == SchemaType.JSON_OBJECT
    
    def compile_schema(self, response_format: Dict[str, Any]) -> str:
        """Convert response_format to GBNF grammar."""
        format_type = response_format.get("type")
        
        if format_type == SchemaType.JSON_OBJECT:
            # Use basic JSON grammar
            return self._get_json_grammar()
        else:
            raise ValueError(f"Currently only json_object format is supported. Got: {format_type}")
    
    def _parse_result(self, raw_output: str, response_format: Dict[str, Any]) -> Any:
        """Parse raw output based on response format."""
        format_type = response_format.get("type")
        
        if format_type in [SchemaType.JSON_OBJECT, SchemaType.JSON_SCHEMA]:
            try:
                # Parse as JSON
                return json.loads(raw_output.strip())
            except json.JSONDecodeError:
                # Return raw output if parsing fails
                return raw_output
        else:
            # For grammar, regex, etc., return raw output
            return raw_output
    
    def _get_json_grammar(self) -> str:
        """Get a basic JSON grammar."""
        return '''
root ::= object
object ::= "{" ws members ws "}"
members ::= member ( "," ws member )*
member ::= string ws ":" ws value
value ::= object | array | string | number | boolean | null
array ::= "[" ws elements ws "]"
elements ::= value ( "," ws value )*
string ::= "\\"" char* "\\""
char ::= [^"\\\\] | "\\\\" escape
escape ::= ["\\\\/bfnrt] | "u" hex hex hex hex
hex ::= [0-9a-fA-F]
number ::= integer ( "." digit+ )? ( [eE] [+-]? digit+ )?
integer ::= "-"? ( "0" | [1-9] digit* )
digit ::= [0-9]
boolean ::= "true" | "false"
null ::= "null"
ws ::= [ \\t\\n\\r]*
'''
    
    def _regex_to_gbnf(self, pattern: str) -> str:
        """
        Convert simple regex pattern to GBNF grammar.
        This is a basic implementation that handles common patterns.
        """
        # Very basic regex to GBNF conversion
        # In practice, this would need much more sophisticated handling
        
        # Handle simple character classes
        if pattern.startswith('^') and pattern.endswith('$'):
            pattern = pattern[1:-1]  # Remove anchors
        
        # Basic patterns
        if pattern == r'\d+':
            return 'root ::= digit+\ndigit ::= [0-9]'
        elif pattern == r'\w+':
            return 'root ::= word+\nword ::= [a-zA-Z0-9_]'
        elif pattern == r'[A-Z][a-z]+':
            return 'root ::= [A-Z] [a-z]+'
        else:
            # Fallback to accepting any string
            return 'root ::= char+\nchar ::= [\\x20-\\x7E]'