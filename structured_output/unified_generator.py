"""
Backend-agnostic structured generation coordinator.
"""
import logging
import platform
from typing import Dict, Any, List, Optional

from .interfaces import StructuredBackend, StructuredRequest, StructuredResponse
from .backends import LlamaCppBackend, MLXBackend, OUTLINES_MLX_AVAILABLE
from .schema_processor import SchemaProcessor, SchemaType
from .converters.gbnf_to_json_schema import convert_gbnf_to_json_schema

logger = logging.getLogger(__name__)


class UnifiedStructuredGenerator:
    """Backend-agnostic structured generation coordinator."""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.schema_processor = SchemaProcessor()
        
        # Initialize available backends
        self.backends: Dict[str, StructuredBackend] = {}
        self._init_backends()
    
    def _init_backends(self):
        """Initialize available backends based on system capabilities."""
        # Always try llama.cpp since it's already in use
        try:
            self.backends["llamacpp"] = LlamaCppBackend(self.model_manager)
            logger.info("Initialized llama.cpp backend for structured output")
        except Exception as e:
            logger.error(f"Failed to initialize llama.cpp backend: {e}")
        
        # Try MLX if on Apple Silicon and outlines is available
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            if OUTLINES_MLX_AVAILABLE:
                try:
                    self.backends["mlx"] = MLXBackend(self.model_manager)
                    logger.info("Initialized MLX backend for structured output")
                except Exception as e:
                    logger.error(f"Failed to initialize MLX backend: {e}")
    
    async def generate(self,
                      prompt: str,
                      response_format: Dict[str, Any],
                      model_tier,
                      **generation_params) -> StructuredResponse:
        """
        Generate structured output using best available backend.
        
        This provides a backend-agnostic interface. If the requested model's backend
        doesn't support the format, we'll transparently use a compatible backend.
        
        Args:
            prompt: The prompt to generate from
            response_format: The response format specification
            model_tier: The model tier to use
            **generation_params: Additional generation parameters
            
        Returns:
            StructuredResponse with generated content
        """
        logger.debug(f"Structured generation request: format_type={response_format.get('type')}, model_tier={model_tier}")
        
        # Validate response format
        if not self.schema_processor.validate_response_format(response_format):
            raise ValueError(f"Invalid response_format: {response_format}")
        
        format_type = response_format.get("type")
        
        # For now, only support json_object mode for consistent behavior
        if format_type != SchemaType.JSON_OBJECT:
            raise ValueError(
                f"Currently only 'json_object' format is supported across all backends. "
                f"Received: {format_type}"
            )
        
        # Get model configuration for requested tier
        model_config = self.model_manager.get_model_config(model_tier)
        if not model_config:
            raise ValueError(f"No model configured for tier: {model_tier}")
        model_backend = model_config.backend.value
        
        # Convert model config to dict
        model_config_dict = {
            'tier': model_tier,
            'path': model_config.path,
            'backend': model_backend,
            'model_name': model_config.path  # For MLX compatibility
        }
        
        # Select the appropriate backend for this model
        if model_backend not in self.backends:
            raise ValueError(f"Backend {model_backend} not available")
        
        selected_backend = self.backends[model_backend]
        
        # Check if we need to convert the schema format
        if not selected_backend.supports_format_type(format_type):
            logger.info(f"Backend {model_backend} doesn't support {format_type}, will convert")
            response_format = self._convert_format(response_format, model_backend)
            format_type = response_format["type"]
        
        logger.info(f"Using {model_tier} tier with {model_backend} backend for {format_type}")
        
        # Create request
        request = StructuredRequest(
            prompt=prompt,
            response_format=response_format,
            model_config=model_config_dict,
            generation_params=generation_params,
            model_tier=model_tier
        )
        
        # Generate
        return await selected_backend.generate(request)
    
    
    def get_supported_formats(self) -> List[str]:
        """Get all supported response format types across all backends."""
        formats = set()
        
        for backend in self.backends.values():
            for format_type in [SchemaType.JSON_OBJECT, SchemaType.JSON_SCHEMA, 
                              SchemaType.GBNF_GRAMMAR, SchemaType.REGEX]:
                if backend.supports_format_type(format_type):
                    formats.add(format_type)
        
        return sorted(list(formats))
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backend names."""
        return list(self.backends.keys())
    
    def _convert_format(self, response_format: Dict[str, Any], target_backend: str) -> Dict[str, Any]:
        """
        Convert response format to one supported by the target backend.
        
        This is the key abstraction - clients can use any format and we'll
        convert it to what the model's backend supports.
        """
        format_type = response_format["type"]
        
        # If target is llama.cpp
        if target_backend == "llamacpp":
            if format_type == "json_schema":
                # Already supported by llama.cpp backend (converts to GBNF)
                return response_format
            elif format_type == "regex":
                # Convert regex to GBNF (basic support)
                # For now, just wrap in GBNF
                return {
                    "type": "gbnf_grammar",
                    "grammar": f'root ::= {response_format["pattern"]}'
                }
            elif format_type == "json_object":
                # Already supported
                return response_format
        
        # If target is MLX
        elif target_backend == "mlx":
            if format_type == "gbnf_grammar":
                # Try to convert GBNF to JSON Schema
                try:
                    logger.info("Converting GBNF grammar to JSON Schema for MLX backend")
                    converted = convert_gbnf_to_json_schema(response_format["grammar"])
                    return converted
                except Exception as e:
                    logger.warning(f"Failed to convert GBNF to JSON Schema: {e}")
                    # Fallback to basic JSON mode
                    return {"type": "json_object"}
            elif format_type == "json_schema":
                # MLX supports JSON Schema natively
                return response_format
            elif format_type == "json_object":
                # MLX supports basic JSON object mode
                return response_format
        
        # If no conversion needed or possible, return as-is
        logger.debug(f"No conversion needed for {format_type} on {target_backend} backend")
        return response_format