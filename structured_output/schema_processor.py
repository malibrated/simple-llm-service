"""
Schema validation and processing utilities.
"""
import json
from enum import Enum
from typing import Dict, Any, Union, Optional


class SchemaType(str, Enum):
    """Supported schema format types."""
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"
    GBNF_GRAMMAR = "gbnf_grammar"
    REGEX = "regex"


class SchemaProcessor:
    """Handles schema validation and normalization."""
    
    def validate_response_format(self, response_format: Dict[str, Any]) -> bool:
        """
        Validate that response_format follows expected structure.
        
        Args:
            response_format: The response format specification
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(response_format, dict):
            return False
            
        format_type = response_format.get("type")
        if not format_type:
            return False
            
        if format_type == SchemaType.JSON_SCHEMA:
            return self._validate_json_schema_format(response_format)
        elif format_type == SchemaType.JSON_OBJECT:
            return True  # Basic JSON object mode
        elif format_type == SchemaType.GBNF_GRAMMAR:
            return "grammar" in response_format
        elif format_type == SchemaType.REGEX:
            return "pattern" in response_format
        
        return False
    
    def _validate_json_schema_format(self, response_format: Dict[str, Any]) -> bool:
        """Validate json_schema response format structure."""
        json_schema = response_format.get("json_schema", {})
        
        # Check required fields according to OpenAI spec
        if not isinstance(json_schema, dict):
            return False
            
        # Name is required
        if "name" not in json_schema:
            return False
            
        # Schema is required and must be a dict
        schema = json_schema.get("schema")
        if not isinstance(schema, dict):
            return False
            
        # Basic JSON Schema validation
        if "type" not in schema:
            return False
            
        return True
    
    def extract_schema(self, response_format: Dict[str, Any]) -> Any:
        """
        Extract the actual schema from response_format.
        
        Args:
            response_format: The response format specification
            
        Returns:
            The extracted schema (dict, string, etc.)
        """
        format_type = response_format["type"]
        
        if format_type == SchemaType.JSON_SCHEMA:
            return response_format["json_schema"]["schema"]
        elif format_type == SchemaType.JSON_OBJECT:
            # Return a basic object schema
            return {"type": "object"}
        elif format_type == SchemaType.GBNF_GRAMMAR:
            return response_format["grammar"]
        elif format_type == SchemaType.REGEX:
            return response_format["pattern"]
        
        raise ValueError(f"Unsupported format type: {format_type}")
    
    def get_schema_type(self, response_format: Dict[str, Any]) -> SchemaType:
        """
        Get the schema type from response_format.
        
        Args:
            response_format: The response format specification
            
        Returns:
            The schema type enum
        """
        format_type = response_format.get("type", "")
        
        try:
            return SchemaType(format_type)
        except ValueError:
            raise ValueError(f"Unknown schema type: {format_type}")
    
    def is_strict_mode(self, response_format: Dict[str, Any]) -> bool:
        """
        Check if strict mode is enabled (OpenAI compatibility).
        
        Args:
            response_format: The response format specification
            
        Returns:
            True if strict mode is enabled
        """
        if response_format.get("type") == SchemaType.JSON_SCHEMA:
            json_schema = response_format.get("json_schema", {})
            return json_schema.get("strict", False)
        
        return False
    
    def get_schema_name(self, response_format: Dict[str, Any]) -> Optional[str]:
        """
        Get the schema name if available.
        
        Args:
            response_format: The response format specification
            
        Returns:
            Schema name or None
        """
        if response_format.get("type") == SchemaType.JSON_SCHEMA:
            json_schema = response_format.get("json_schema", {})
            return json_schema.get("name")
        
        return None