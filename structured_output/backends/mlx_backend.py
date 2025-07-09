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

from ..interfaces import StructuredBackend, StructuredRequest, StructuredResponse

# Export OUTLINES_MLX_AVAILABLE for compatibility
OUTLINES_MLX_AVAILABLE = MLX_AVAILABLE


class MLXBackend(StructuredBackend):
    """MLX backend using Outlines for constrained generation"""
    
    def __init__(self, model_manager):
        if not MLX_AVAILABLE:
            raise ImportError("MLX support not available. Install with: pip install mlx-lm")
        
        self.model_manager = model_manager
        self._generation_lock = None  # Will be set to the model manager's MLX lock
    
    def generate_sync(self, prompt: str, model_wrapper, **kwargs) -> str:
        """Synchronous generation helper to avoid nested async issues."""
        logger.info(f"[TRACE] generate_sync called with kwargs: {list(kwargs.keys())}")
        
        # If prompt ends with "Assistant:", we need to add JSON instruction right after it
        if prompt.endswith("Assistant:"):
            json_prompt = prompt + " " + "{"
        else:
            # Build JSON prompt
            json_prompt = f"""{prompt}

Respond with valid JSON only. Start with {{ and end with }}. No markdown or backticks."""
        
        # Use the model wrapper's synchronous generate internals
        if hasattr(model_wrapper, 'model') and hasattr(model_wrapper, 'tokenizer'):
            # This is an MLXWrapper, use its internal generate method
            
            # Extract parameters
            temp = kwargs.get('temperature', 0.1)
            max_tokens = kwargs.get('max_tokens', 200)
            top_p = kwargs.get('top_p', 0.95)
            top_k = kwargs.get('top_k', 0)
            
            # Create sampler
            sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
            
            # Generate directly
            response = mlx_generate(
                model_wrapper.model,
                model_wrapper.tokenizer,
                json_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False
            )
            
            # Extract JSON from markdown if needed
            import re
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1).strip()
            
            return response
        else:
            raise ValueError("Invalid model wrapper for MLX generation")
    
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
                        return self.generate_sync(request.prompt, model_wrapper, **all_gen_params)
                    
                    logger.info(f"[TRACE] Calling run_in_executor with sync_wrapper")
                    result = await loop.run_in_executor(None, sync_wrapper)
                    logger.info(f"[TRACE] generate_sync returned: {type(result)}")
            else:
                # No lock, run directly
                logger.info("[TRACE] No lock, calling generate_sync directly")
                result = self.generate_sync(request.prompt, model_wrapper, **all_gen_params)
            
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
                    parsed_result = json.loads(content)
                    content = json.dumps(parsed_result)  # Normalize
                except json.JSONDecodeError:
                    # If we can't parse it, return a simple object
                    logger.warning(f"Could not parse JSON from MLX response: {content[:100]}")
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