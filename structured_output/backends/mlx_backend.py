# structured_output/backends/mlx_backend.py
import time
import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    from outlines import models, generate
    from mlx_lm import load as mlx_load
    OUTLINES_MLX_AVAILABLE = True
except ImportError:
    OUTLINES_MLX_AVAILABLE = False
    models = None
    generate = None
    mlx_load = None

from ..interfaces import StructuredBackend, StructuredRequest, StructuredResponse


class MLXBackend(StructuredBackend):
    """MLX backend using Outlines for constrained generation"""
    
    def __init__(self, model_manager):
        if not OUTLINES_MLX_AVAILABLE:
            raise ImportError("Outlines MLX support not available. Install with: pip install outlines mlx-lm")
        
        self.model_manager = model_manager
        self._loaded_models = {}  # Cache loaded Outlines models
        self._generators = {}     # Cache compiled generators
    
    async def generate(self, request: StructuredRequest) -> StructuredResponse:
        start_time = time.perf_counter()
        
        # Get model path from request
        model_path = request.model_config.get("path")
        if not model_path:
            raise ValueError("No model path provided for MLX backend")
        
        # Load and wrap model if not cached
        if model_path not in self._loaded_models:
            logger.info(f"Loading MLX model from {model_path} for Outlines")
            
            # Load MLX model and tokenizer
            mlx_model, tokenizer = mlx_load(model_path)
            
            # Use the correct API: from_mlxlm
            self._loaded_models[model_path] = models.from_mlxlm(mlx_model, tokenizer)
            logger.info("MLX model wrapped with Outlines successfully")
        
        model = self._loaded_models[model_path]
        
        # Get or create generator
        format_type = request.response_format["type"]
        generator_key = f"{model_path}_{format_type}"
        
        if generator_key not in self._generators:
            if format_type == "json_object":
                # For basic JSON object, create a flexible Pydantic model
                from pydantic import BaseModel
                
                # Create a model that accepts any fields
                class FlexibleModel(BaseModel):
                    class Config:
                        extra = 'allow'  # This allows any additional fields
                
                # Create JSON generator with the flexible schema
                self._generators[generator_key] = generate.json(model, FlexibleModel)
                logger.info("Created JSON generator for MLX model with flexible schema")
            else:
                raise ValueError(f"Unsupported format type for MLX: {format_type}")
        
        generator = self._generators[generator_key]
        
        # Generate with constraints
        # Extract generation parameters
        gen_params = {}
        if "max_tokens" in request.generation_params:
            gen_params["max_tokens"] = request.generation_params["max_tokens"]
        # Note: Outlines handles temperature internally, we don't need to pass it
        
        logger.info(f"Generating with Outlines MLX backend, params: {gen_params}")
        
        try:
            # Generate structured output
            result = generator(request.prompt, **gen_params)
            
            # The result is a Pydantic model instance
            if hasattr(result, 'model_dump'):
                # Pydantic v2
                parsed_result = result.model_dump()
            elif hasattr(result, 'dict'):
                # Pydantic v1
                parsed_result = result.dict()
            else:
                # This shouldn't happen, but fallback
                parsed_result = {}
            
            # Convert to clean JSON string
            content = json.dumps(parsed_result)
            
            logger.info(f"Successfully generated JSON: {content[:100]}...")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Fallback to empty object
            content = "{}"
            parsed_result = {}
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return StructuredResponse(
            content=content,
            parsed_result=parsed_result,
            processing_time_ms=processing_time,
            backend_used="mlx-outlines"
        )
    
    def supports_format_type(self, format_type: str) -> bool:
        # For now, only support json_object for consistent behavior
        return format_type == "json_object"
    
    def compile_schema(self, response_format: Dict[str, Any]) -> str:
        """MLX doesn't need explicit compilation"""
        return f"mlx_outlines_{hash(str(response_format))}"