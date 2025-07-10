# structured_output/backends/mlx_backend.py
import time
import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    from mlx_lm import load as mlx_load
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Failed to import MLX dependencies: {e}")
    MLX_AVAILABLE = False
    mlx_load = None
    mlx_generate = None
    make_sampler = None

try:
    import outlines
    from outlines import models, generate
    OUTLINES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Failed to import Outlines: {e}")
    OUTLINES_AVAILABLE = False
    outlines = None
    models = None
    generate = None

from ..interfaces import StructuredBackend, StructuredRequest, StructuredResponse

# Export OUTLINES_MLX_AVAILABLE for compatibility
OUTLINES_MLX_AVAILABLE = MLX_AVAILABLE and OUTLINES_AVAILABLE


class MLXBackend(StructuredBackend):
    """MLX backend using Outlines for constrained generation"""
    
    def __init__(self, model_manager):
        if not MLX_AVAILABLE:
            raise ImportError("MLX support not available. Install with: pip install mlx-lm")
        if not OUTLINES_AVAILABLE:
            raise ImportError("Outlines not available. Install with: pip install outlines")
        
        self.model_manager = model_manager
        self._generation_lock = None  # Will be set to the model manager's MLX lock
        self._outlines_models = {}  # Cache for Outlines model wrappers
    
    def _get_or_create_outlines_model(self, model_wrapper):
        """Get or create an Outlines model wrapper for the MLX model."""
        model_id = id(model_wrapper)
        
        if model_id not in self._outlines_models:
            # Create Outlines model from MLX model
            logger.info("[TRACE] Creating Outlines wrapper for MLX model")
            self._outlines_models[model_id] = models.from_mlxlm(model_wrapper.model, model_wrapper.tokenizer)
            logger.info("[TRACE] Outlines wrapper created successfully")
        
        return self._outlines_models[model_id]
    
    def generate_sync(self, prompt: str, model_wrapper, response_format: dict, **kwargs) -> str:
        """Synchronous generation using Outlines for constrained JSON."""
        logger.info(f"[TRACE] generate_sync called with response_format: {response_format}")
        
        # Get or create the Outlines model wrapper
        outlines_model = self._get_or_create_outlines_model(model_wrapper)
        
        # For json_object format, use Outlines JSON generation
        if response_format.get("type") == "json_object":
            logger.info("[TRACE] Using Outlines JSON generation")
            
            # Add a helpful hint about JSON generation if not already present
            if "json" not in prompt.lower() and "Assistant:" in prompt:
                prompt = prompt.replace("Assistant:", "Assistant: I'll generate the requested data as a JSON object.")
            elif "json" not in prompt.lower():
                prompt += "\n\nGenerate the requested data as a JSON object."
            
            # Create a generic JSON schema that accepts any valid JSON object
            # Using a simple schema that allows flexible JSON structure
            schema = {
                "type": "object",
                "additionalProperties": True  # Allow any properties
            }
            
            # Create JSON generator
            try:
                json_generator = generate.json(outlines_model, schema)
                logger.info("[TRACE] Created Outlines JSON generator")
                
                # Generate with the constrained generator
                response = json_generator(prompt)
                logger.info(f"[TRACE] Outlines generated: {str(response)[:200]}...")
                
                # Convert to JSON string if it's a dict
                if isinstance(response, dict):
                    return json.dumps(response)
                return str(response)
                
            except Exception as e:
                logger.error(f"[TRACE] Outlines generation failed: {e}")
                # Fall back to basic generation
                logger.info("[TRACE] Falling back to basic MLX generation")
                
                # Add JSON instruction to prompt
                json_prompt = f"{prompt}\n\nRespond with valid JSON only:"
                
                # Use basic MLX generation
                temp = kwargs.get('temperature', 0.1)
                max_tokens = kwargs.get('max_tokens', 2048)
                
                response = outlines_model.generate(
                    json_prompt,
                    max_tokens=max_tokens,
                    temperature=temp
                )
                
                return response
        else:
            raise ValueError(f"Unsupported response format type: {response_format.get('type')}")
    
    async def generate(self, request: StructuredRequest) -> StructuredResponse:
        start_time = time.perf_counter()
        
        # Get model path from request
        model_path = request.model_config.get("path")
        if not model_path:
            raise ValueError("No model path provided for MLX backend")
        
        # Get the MLX lock from model manager if available
        if self._generation_lock is None and hasattr(request, 'model_tier'):
            # Try to get the lock from model manager
            if request.model_tier in self.model_manager._mlx_locks:
                self._generation_lock = self.model_manager._mlx_locks[request.model_tier]
        
        # Get model wrapper directly from model manager
        logger.info(f"[TRACE] Getting model wrapper from model manager")
        if not hasattr(request, 'model_tier') or request.model_tier not in self.model_manager.models:
            raise ValueError(f"Model tier {request.model_tier} not loaded in model manager")
        
        model_wrapper = self.model_manager.models[request.model_tier]
        logger.info(f"[TRACE] Got model wrapper: {type(model_wrapper)}")
        
        # For MLX, we'll use direct generation
        format_type = request.response_format["type"]
        
        if format_type != "json_object":
            raise ValueError(f"Unsupported format type for MLX: {format_type}")
        
        # Generate with constraints
        # Store all generation parameters for the wrapper
        all_gen_params = request.generation_params.copy()
        
        logger.info(f"Generating with Outlines MLX backend, params: {all_gen_params}")
        
        try:
            # Use async lock if available
            if self._generation_lock:
                async with self._generation_lock:
                    logger.info("[TRACE] Inside MLX lock, calling generate_sync")
                    # Run synchronous generation in executor to avoid blocking
                    import asyncio
                    loop = asyncio.get_event_loop()
                    
                    # Create a wrapper function that takes no arguments
                    logger.info(f"[TRACE] Creating wrapper function with gen_params: {list(all_gen_params.keys())}")
                    def sync_wrapper():
                        return self.generate_sync(request.prompt, model_wrapper, request.response_format, **all_gen_params)
                    
                    logger.info(f"[TRACE] Calling run_in_executor with sync_wrapper")
                    result = await loop.run_in_executor(None, sync_wrapper)
                    logger.info(f"[TRACE] generate_sync returned: {type(result)}")
            else:
                # No lock, run directly
                logger.info("[TRACE] No lock, calling generate_sync directly")
                result = self.generate_sync(request.prompt, model_wrapper, request.response_format, **all_gen_params)
            
            # Handle different result types
            if isinstance(result, str):
                # Direct string result (from fallback generator)
                content = result
                # Try to extract JSON from the response
                import re
                # Look for JSON between triple backticks or just parse directly
                json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', result, re.DOTALL)
                if json_match:
                    content = json_match.group(1).strip()
                
                # Try to parse the content as JSON
                try:
                    # First, try to fix common JSON issues
                    content_cleaned = content.strip()
                    
                    # Replace smart quotes with regular quotes
                    content_cleaned = content_cleaned.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
                    
                    # If it doesn't start with {, try to add it
                    if not content_cleaned.startswith('{'):
                        # Check if it looks like JSON content without the opening brace
                        if content_cleaned.strip().startswith('"') or 'name' in content_cleaned[:50]:
                            content_cleaned = '{' + content_cleaned
                        else:
                            # Find the first { if there is one
                            brace_idx = content_cleaned.find('{')
                            if brace_idx != -1:
                                content_cleaned = content_cleaned[brace_idx:]
                    
                    # If it's incomplete (doesn't end with }), try to complete it
                    if content_cleaned and not content_cleaned.rstrip().endswith('}'):
                        # Count braces to see how many we need
                        open_braces = content_cleaned.count('{')
                        close_braces = content_cleaned.count('}')
                        missing_braces = open_braces - close_braces
                        
                        # Add missing quotes if needed
                        if content_cleaned.rstrip().endswith('"'):
                            content_cleaned += '}' * missing_braces
                        else:
                            # Try to close any open string
                            if '"' in content_cleaned and not content_cleaned.rstrip().endswith('"'):
                                content_cleaned += '"'
                            content_cleaned += '}' * missing_braces
                    
                    parsed_result = json.loads(content_cleaned)
                    content = json.dumps(parsed_result)  # Normalize
                except json.JSONDecodeError as e:
                    # If we still can't parse it, return a simple object
                    logger.warning(f"Could not parse JSON from MLX response: {content[:200]}... Error: {e}")
                    parsed_result = {"error": "Failed to generate valid JSON", "raw": content}
                    content = json.dumps(parsed_result)
            elif hasattr(result, 'model_dump'):
                # Pydantic v2
                parsed_result = result.model_dump()
                content = json.dumps(parsed_result)
            elif hasattr(result, 'dict'):
                # Pydantic v1
                parsed_result = result.dict()
                content = json.dumps(parsed_result)
            else:
                # Unknown result type
                logger.warning(f"Unknown result type from generator: {type(result)}")
                parsed_result = {"error": "Unknown result type", "value": str(result)}
                content = json.dumps(parsed_result)
            
            logger.info(f"Successfully generated JSON: {content[:100]}...")
            
        except Exception as e:
            logger.error(f"[TRACE] Generation failed: {e}")
            import traceback
            tb = traceback.format_exc()
            logger.error(f"[TRACE] Full traceback:\n{tb}")
            
            # Check if this is the run_in_executor error
            if "run_in_executor" in str(e):
                logger.error("[TRACE] This is the run_in_executor error!")
                # Log the current stack to see where we are
                import inspect
                frames = inspect.getouterframes(inspect.currentframe())
                logger.error("[TRACE] Call stack:")
                for frame in frames[:10]:  # First 10 frames
                    logger.error(f"  {frame.filename}:{frame.lineno} in {frame.function}")
            
            # For debugging, include more info in the error
            error_info = {
                "error": str(e),
                "type": type(e).__name__,
                "traceback": tb.split('\n')  # Full traceback
            }
            content = json.dumps(error_info)
            parsed_result = error_info
        
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