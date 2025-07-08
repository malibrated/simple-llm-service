"""
JSON Schema to GBNF (GGML BNF) converter for llama.cpp.
"""
import json
from typing import Dict, Any, List, Set, Optional


class JSONSchemaToGBNF:
    """Converts JSON Schema to GBNF grammar for llama.cpp."""
    
    def __init__(self):
        self.rules: List[str] = []
        self.rule_names: Set[str] = set()
        self.rule_counter = 0
        
    def convert(self, json_schema: Dict[str, Any]) -> str:
        """
        Convert JSON Schema to GBNF grammar.
        
        Args:
            json_schema: JSON Schema dictionary
            
        Returns:
            GBNF grammar string
        """
        self.rules = []
        self.rule_names = set()
        self.rule_counter = 0
        
        # Generate root rule
        self._generate_rule(json_schema, "root")
        
        # Add common utility rules
        self._add_common_rules()
        
        return '\n'.join(self.rules)
    
    def _generate_rule(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Generate GBNF rule for a schema element."""
        if rule_name in self.rule_names:
            return rule_name
            
        self.rule_names.add(rule_name)
        
        schema_type = schema.get('type', 'string')
        
        if isinstance(schema_type, list):
            # Handle multiple types (union)
            return self._generate_union_rule(schema, rule_name, schema_type)
        
        if schema_type == 'object':
            return self._generate_object_rule(schema, rule_name)
        elif schema_type == 'array':
            return self._generate_array_rule(schema, rule_name)
        elif schema_type == 'string':
            return self._generate_string_rule(schema, rule_name)
        elif schema_type == 'number':
            return self._generate_number_rule(schema, rule_name)
        elif schema_type == 'integer':
            return self._generate_integer_rule(schema, rule_name)
        elif schema_type == 'boolean':
            return self._generate_boolean_rule(schema, rule_name)
        elif schema_type == 'null':
            return self._generate_null_rule(schema, rule_name)
        else:
            # Default to string for unknown types
            return self._generate_string_rule(schema, rule_name)
    
    def _generate_object_rule(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Generate rule for object type."""
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        additional_props = schema.get('additionalProperties', True)
        
        if not properties and additional_props:
            # Empty object or object with any properties
            self.rules.append(f'{rule_name} ::= "{{" ws "}}" | "{{" ws members ws "}}"')
            if 'members' not in self.rule_names:
                self.rules.append('members ::= member ( ws "," ws member )*')
                self.rules.append('member ::= string ws ":" ws value')
                self.rule_names.add('members')
                self.rule_names.add('member')
            return rule_name
        
        if not properties:
            # Empty object only
            self.rules.append(f'{rule_name} ::= "{{" ws "}}"')
            return rule_name
        
        # Generate rules for each property
        property_rules = []
        
        # Required properties first
        for prop_name in required:
            if prop_name in properties:
                prop_schema = properties[prop_name]
                prop_rule_name = f"{rule_name}_{self._safe_name(prop_name)}"
                self._generate_rule(prop_schema, prop_rule_name)
                property_rules.append((prop_name, prop_rule_name, True))
        
        # Optional properties
        for prop_name, prop_schema in properties.items():
            if prop_name not in required:
                prop_rule_name = f"{rule_name}_{self._safe_name(prop_name)}"
                self._generate_rule(prop_schema, prop_rule_name)
                property_rules.append((prop_name, prop_rule_name, False))
        
        # Build object rule
        if len(property_rules) == 1 and property_rules[0][2]:  # Single required property
            prop_name, prop_rule, _ = property_rules[0]
            self.rules.append(
                f'{rule_name} ::= "{{" ws "\\"" "{prop_name}" "\\"" ws ":" ws {prop_rule} ws "}}"'
            )
        else:
            # Multiple properties or optional properties
            members_parts = []
            
            # Add required properties
            req_parts = []
            for prop_name, prop_rule, is_req in property_rules:
                if is_req:
                    req_parts.append(f'"\\"" "{prop_name}" "\\"" ws ":" ws {prop_rule}')
            
            if req_parts:
                members_parts.append(' ws "," ws '.join(req_parts))
            
            # Add optional properties
            opt_parts = []
            for prop_name, prop_rule, is_req in property_rules:
                if not is_req:
                    opt_part = f'( ws "," ws "\\"" "{prop_name}" "\\"" ws ":" ws {prop_rule} )?'
                    opt_parts.append(opt_part)
            
            # Combine all parts
            if members_parts and opt_parts:
                members_expr = members_parts[0] + ' '.join(opt_parts)
            elif members_parts:
                members_expr = members_parts[0]
            elif opt_parts:
                # All properties are optional
                # Build optional properties expression
                opt_items = []
                for p in property_rules:
                    opt_items.append(f'"\\"" "{p[0]}" "\\"" ws ":" ws {p[1]}')
                members_expr = f'( {" | ".join(opt_items)} )?'
            else:
                members_expr = ''
            
            self.rules.append(f'{rule_name} ::= "{{" ws {members_expr} ws "}}"')
        
        return rule_name
    
    def _generate_array_rule(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Generate rule for array type."""
        items_schema = schema.get('items', {'type': 'string'})
        min_items = schema.get('minItems', 0)
        max_items = schema.get('maxItems')
        
        items_rule_name = f"{rule_name}_item"
        self._generate_rule(items_schema, items_rule_name)
        
        if min_items == 0 and max_items is None:
            # Any number of items
            self.rules.append(
                f'{rule_name} ::= "[" ws "]" | "[" ws {items_rule_name} ( ws "," ws {items_rule_name} )* ws "]"'
            )
        elif min_items == 1 and max_items is None:
            # At least one item
            self.rules.append(
                f'{rule_name} ::= "[" ws {items_rule_name} ( ws "," ws {items_rule_name} )* ws "]"'
            )
        elif min_items == max_items:
            # Exact number of items
            items_list = [items_rule_name] * min_items
            items_expr = ' ws "," ws '.join(items_list)
            self.rules.append(f'{rule_name} ::= "[" ws {items_expr} ws "]"')
        else:
            # Complex case - use simple array for now
            self.rules.append(
                f'{rule_name} ::= "[" ws "]" | "[" ws {items_rule_name} ( ws "," ws {items_rule_name} )* ws "]"'
            )
        
        return rule_name
    
    def _generate_string_rule(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Generate rule for string type."""
        enum_values = schema.get('enum')
        pattern = schema.get('pattern')
        min_length = schema.get('minLength')
        max_length = schema.get('maxLength')
        
        if enum_values:
            # Enum constraint
            enum_options = ' | '.join(f'"\\"" "{value}" "\\""' for value in enum_values)
            self.rules.append(f'{rule_name} ::= {enum_options}')
        elif pattern:
            # Pattern constraint - fallback to basic string
            # TODO: Convert regex to GBNF if possible
            self.rules.append(f'{rule_name} ::= string')
        else:
            # Basic string
            self.rules.append(f'{rule_name} ::= string')
        
        return rule_name
    
    def _generate_number_rule(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Generate rule for number type."""
        # TODO: Handle minimum, maximum, multipleOf constraints
        self.rules.append(f'{rule_name} ::= number')
        return rule_name
    
    def _generate_integer_rule(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Generate rule for integer type."""
        # TODO: Handle minimum, maximum, multipleOf constraints
        self.rules.append(f'{rule_name} ::= integer')
        return rule_name
    
    def _generate_boolean_rule(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Generate rule for boolean type."""
        self.rules.append(f'{rule_name} ::= "true" | "false"')
        return rule_name
    
    def _generate_null_rule(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Generate rule for null type."""
        self.rules.append(f'{rule_name} ::= "null"')
        return rule_name
    
    def _generate_union_rule(self, schema: Dict[str, Any], rule_name: str, types: List[str]) -> str:
        """Generate rule for union types."""
        sub_rules = []
        
        for type_name in types:
            sub_rule_name = f"{rule_name}_{type_name}"
            sub_schema = {'type': type_name}
            # Copy other constraints
            for key in ['enum', 'pattern', 'properties', 'items']:
                if key in schema:
                    sub_schema[key] = schema[key]
            
            self._generate_rule(sub_schema, sub_rule_name)
            sub_rules.append(sub_rule_name)
        
        self.rules.append(f'{rule_name} ::= {" | ".join(sub_rules)}')
        return rule_name
    
    def _safe_name(self, name: str) -> str:
        """Convert property name to safe rule name."""
        # Replace special characters with underscores
        safe = ''.join(c if c.isalnum() else '_' for c in name)
        # Ensure it starts with a letter
        if safe and not safe[0].isalpha():
            safe = f'prop_{safe}'
        return safe
    
    def _add_common_rules(self):
        """Add common utility rules used by all grammars."""
        common_rules = [
            # Whitespace
            'ws ::= [ \\t\\n\\r]*',
            
            # String
            'string ::= "\\"" char* "\\""',
            'char ::= [^"\\\\] | "\\\\" escape',
            'escape ::= ["\\\\/bfnrt] | "u" hex hex hex hex',
            'hex ::= [0-9a-fA-F]',
            
            # Number
            'number ::= integer ( "." digit+ )? ( [eE] [+-]? digit+ )?',
            'integer ::= "-"? ( "0" | [1-9] digit* )',
            'digit ::= [0-9]',
            
            # Generic value (for dynamic objects)
            'value ::= string | number | boolean | null | object | array',
            'object ::= "{" ws "}"',
            'array ::= "[" ws "]"',
            'boolean ::= "true" | "false"',
            'null ::= "null"'
        ]
        
        # Only add rules that haven't been defined
        for rule in common_rules:
            rule_name = rule.split(' ::= ')[0]
            if rule_name not in self.rule_names:
                self.rules.append(rule)
                self.rule_names.add(rule_name)