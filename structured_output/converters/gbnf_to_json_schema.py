"""
GBNF Grammar to JSON Schema converter.

This is a simplified converter that handles basic GBNF patterns.
Full GBNF parsing would require a proper grammar parser.
"""
import re
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class GBNFToJSONSchemaConverter:
    """Convert GBNF grammar to JSON Schema (simplified)."""
    
    def convert(self, gbnf_grammar: str) -> Dict[str, Any]:
        """
        Convert GBNF grammar to JSON Schema.
        
        This is a simplified implementation that handles common patterns.
        A full implementation would require a proper GBNF parser.
        """
        # Parse the grammar into rules
        rules = self._parse_rules(gbnf_grammar)
        
        if not rules:
            raise ValueError("No valid rules found in GBNF grammar")
        
        # Start from root rule
        if "root" not in rules:
            raise ValueError("No 'root' rule found in GBNF grammar")
        
        # Try to infer schema from root rule
        schema = self._rule_to_schema(rules["root"], rules)
        
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "gbnf_converted",
                "strict": False,  # GBNF conversions may not be strict
                "schema": schema
            }
        }
    
    def _parse_rules(self, grammar: str) -> Dict[str, str]:
        """Parse GBNF grammar into rule dictionary."""
        rules = {}
        
        # Split by lines and process each rule
        lines = grammar.strip().split('\n')
        current_rule = None
        current_definition = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Check if this is a new rule definition
            if '::=' in line:
                # Save previous rule if exists
                if current_rule:
                    rules[current_rule] = ' '.join(current_definition)
                
                # Parse new rule
                parts = line.split('::=', 1)
                current_rule = parts[0].strip()
                current_definition = [parts[1].strip()]
            else:
                # Continuation of current rule
                if current_rule:
                    current_definition.append(line)
        
        # Save last rule
        if current_rule:
            rules[current_rule] = ' '.join(current_definition)
        
        return rules
    
    def _rule_to_schema(self, rule_def: str, all_rules: Dict[str, str]) -> Dict[str, Any]:
        """Convert a rule definition to JSON Schema."""
        rule_def = rule_def.strip()
        
        # Handle object pattern: {...}
        if rule_def.startswith('{') and rule_def.endswith('}'):
            return self._parse_object_pattern(rule_def, all_rules)
        
        # Handle array pattern: [...]
        if rule_def.startswith('[') and rule_def.endswith(']'):
            return {"type": "array", "items": {"type": "string"}}
        
        # Handle string literal
        if rule_def.startswith('"') and rule_def.endswith('"'):
            return {"type": "string", "const": rule_def[1:-1]}
        
        # Handle references to other rules
        if rule_def in all_rules:
            return self._rule_to_schema(all_rules[rule_def], all_rules)
        
        # Handle common patterns
        if 'string' in rule_def:
            return {"type": "string"}
        elif 'number' in rule_def:
            return {"type": "number"}
        elif 'boolean' in rule_def:
            return {"type": "boolean"}
        elif 'null' in rule_def:
            return {"type": "null"}
        elif 'object' in rule_def:
            return {"type": "object"}
        elif 'array' in rule_def:
            return {"type": "array"}
        
        # Check for alternation (|)
        if '|' in rule_def:
            options = [opt.strip() for opt in rule_def.split('|')]
            # If all options are string literals, create enum
            if all(opt.startswith('"') and opt.endswith('"') for opt in options):
                return {
                    "type": "string",
                    "enum": [opt[1:-1] for opt in options]
                }
            # Otherwise create anyOf
            schemas = []
            for opt in options:
                try:
                    schema = self._rule_to_schema(opt, all_rules)
                    schemas.append(schema)
                except:
                    pass
            if schemas:
                return {"anyOf": schemas}
        
        # Default to string
        logger.warning(f"Could not determine type for rule: {rule_def}, defaulting to string")
        return {"type": "string"}
    
    def _parse_object_pattern(self, pattern: str, all_rules: Dict[str, str]) -> Dict[str, Any]:
        """Parse an object pattern from GBNF."""
        # This is simplified - real GBNF objects are more complex
        properties = {}
        required = []
        
        # Try to extract property patterns
        # Look for patterns like "\"name\":" or similar
        prop_pattern = r'"([^"]+)":\s*(\w+)'
        matches = re.findall(prop_pattern, pattern)
        
        for prop_name, prop_type in matches:
            required.append(prop_name)
            
            # Try to resolve the property type
            if prop_type in all_rules:
                properties[prop_name] = self._rule_to_schema(all_rules[prop_type], all_rules)
            elif prop_type == "string":
                properties[prop_name] = {"type": "string"}
            elif prop_type == "number":
                properties[prop_name] = {"type": "number"}
            elif prop_type == "boolean":
                properties[prop_name] = {"type": "boolean"}
            else:
                properties[prop_name] = {"type": "string"}
        
        schema = {"type": "object"}
        if properties:
            schema["properties"] = properties
        if required:
            schema["required"] = required
            
        return schema


def convert_gbnf_to_json_schema(gbnf_grammar: str) -> Dict[str, Any]:
    """Convert GBNF grammar to JSON Schema format."""
    converter = GBNFToJSONSchemaConverter()
    return converter.convert(gbnf_grammar)