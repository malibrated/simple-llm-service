# structured_output/backends/mlx_backend.py
import time
import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    from mlx_lm import load as mlx_load
    from outlines.models import from_mlxlm
    from outlines import generate
    from pydantic import BaseModel, create_model
    import outlines
    OUTLINES_MLX_AVAILABLE = True
    logger.info(f"Outlines version: {outlines.__version__ if hasattr(outlines, '__version__') else 'unknown'}")
except ImportError as e:
    logger.warning(f"Failed to import Outlines MLX dependencies: {e}")
    OUTLINES_MLX_AVAILABLE = False
    mlx_load = None
    from_mlxlm = None
    generate = None
    BaseModel = None
    create_model = None

from ..interfaces import StructuredBackend, StructuredRequest, StructuredResponse


class MLXBackend(StructuredBackend):
    """MLX backend using Outlines for constrained generation"""
    
    def __init__(self, model_manager):
        if not OUTLINES_MLX_AVAILABLE:
            raise ImportError("Outlines MLX support not available. Install with: pip install outlines mlx-lm")
        
        self.model_manager = model_manager
        self._loaded_models = {}  # Cache loaded Outlines models
        self._generators = {}     # Cache compiled generators
        self._generation_lock = None  # Will be set to the model manager's MLX lock
    
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
        
        # Check if model manager already has this model loaded
        mlx_model = None
        tokenizer = None
        
        if hasattr(request, 'model_tier') and request.model_tier in self.model_manager.models:
            model_wrapper = self.model_manager.models[request.model_tier]
            if hasattr(model_wrapper, 'model') and hasattr(model_wrapper, 'tokenizer'):
                logger.info(f"Using existing MLX model from model manager for {request.model_tier.value}")
                mlx_model = model_wrapper.model
                tokenizer = model_wrapper.tokenizer
        
        # Load and wrap model if not cached and not from model manager
        if model_path not in self._loaded_models:
            if mlx_model is not None and tokenizer is not None:
                # Use model from model manager
                logger.info("Reusing MLX model from model manager")
                try:
                    # Wrap with Outlines
                    wrapped_model = from_mlxlm(mlx_model, tokenizer)
                    self._loaded_models[model_path] = wrapped_model
                    logger.info("Successfully wrapped MLX model with Outlines")
                except Exception as e:
                    logger.error(f"Failed to wrap MLX model with Outlines: {e}")
                    raise
            else:
                # Load model ourselves (fallback)
                logger.warning(f"Loading separate MLX model from {model_path} for Outlines")
                mlx_model, tokenizer = mlx_load(model_path)
                
                try:
                    # Wrap with Outlines
                    wrapped_model = from_mlxlm(mlx_model, tokenizer)
                    self._loaded_models[model_path] = wrapped_model
                    logger.info("Successfully loaded and wrapped MLX model with Outlines")
                except Exception as e:
                    logger.error(f"Failed to wrap MLX model with Outlines: {e}")
                    raise
            
            logger.info("MLX model ready for structured generation")
        
        model = self._loaded_models[model_path]
        
        # Get or create generator
        format_type = request.response_format["type"]
        generator_key = f"{model_path}_{format_type}"
        
        if generator_key not in self._generators:
            if format_type == "json_object":
                # For MLX, use direct generation with JSON prompting
                # This avoids the complexity of Outlines integration issues
                def mlx_json_generator(prompt, **kwargs):
                    logger.info(f"mlx_json_generator called with kwargs: {list(kwargs.keys())}")
                    
                    # Build a JSON-focused prompt
                    json_prompt = f"""{prompt}

IMPORTANT: You must respond with valid JSON only. 
- Do NOT include any markdown formatting or code blocks
- Do NOT include backticks (```) 
- Start your response with {{ and end with }}
- Ensure the JSON is properly formatted and valid"""
                    
                    try:
                        # For direct generation, we need to use the underlying MLX model
                        if hasattr(model, 'model') and hasattr(model, 'mlx_tokenizer'):
                            logger.info("Using direct MLX generation")
                            from mlx_lm import generate as mlx_generate
                            from mlx_lm.sample_utils import make_sampler
                            
                            # Extract ONLY the parameters that go to make_sampler
                            temp = kwargs.get('temperature', 0.1)
                            top_p = kwargs.get('top_p', 0.95)
                            top_k = kwargs.get('top_k', 0)
                            
                            # Extract parameters that go directly to generate
                            max_tokens = kwargs.get('max_tokens', 200)
                            
                            # Create sampler with the sampling parameters
                            sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
                            
                            logger.info(f"Calling mlx_generate with max_tokens={max_tokens} and sampler")
                            
                            # Generate with MLX directly
                            # Only pass prompt, max_tokens, and sampler to generate
                            response = mlx_generate(
                                model.model,
                                model.mlx_tokenizer,
                                json_prompt,
                                max_tokens=max_tokens,
                                sampler=sampler,
                                verbose=False
                            )
                            logger.info("mlx_generate completed successfully")
                            
                            # Post-process to extract JSON if wrapped in markdown
                            import re
                            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
                            if json_match:
                                response = json_match.group(1).strip()
                                logger.info("Extracted JSON from markdown wrapper")
                            
                            return response
                        else:
                            # This must be an Outlines-wrapped model
                            logger.warning("Using Outlines model.generate")
                            logger.info(f"Model type: {type(model)}")
                            
                            # The Outlines MLXLM model expects different parameters
                            # It forwards kwargs to mlx_lm.generate, which only accepts certain params
                            # Only pass parameters that MLX accepts
                            max_tokens = kwargs.get('max_tokens', 200)
                            
                            # MLX doesn't accept temperature/top_p/top_k directly, needs sampler
                            # But Outlines doesn't expose sampler parameter
                            # So we can only pass max_tokens
                            logger.info(f"Calling Outlines model.generate with max_tokens={max_tokens}")
                            response = model.generate(json_prompt, max_tokens=max_tokens)
                            logger.info("model.generate completed")
                            
                            # Post-process to extract JSON if wrapped in markdown
                            import re
                            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
                            if json_match:
                                response = json_match.group(1).strip()
                                logger.info("Extracted JSON from markdown wrapper")
                            
                            return response
                    except Exception as e:
                        logger.error(f"MLX JSON generation failed: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        raise
                
                self._generators[generator_key] = mlx_json_generator
                logger.info("Created direct MLX JSON generator")
            else:
                raise ValueError(f"Unsupported format type for MLX: {format_type}")
        
        generator = self._generators[generator_key]
        
        # Generate with constraints
        # Store all generation parameters for the wrapper
        all_gen_params = request.generation_params.copy()
        
        logger.info(f"Generating with Outlines MLX backend, params: {all_gen_params}")
        
        try:
            # Generate structured output with lock if available
            logger.info(f"Calling generator with params: {list(all_gen_params.keys())}")
            logger.info(f"Generator type: {type(generator)}")
            logger.info(f"Generator is: {generator}")
            
            if self._generation_lock:
                # Use async lock to serialize MLX access
                async with self._generation_lock:
                    # Just call the generator synchronously
                    logger.info("Inside lock, calling generator")
                    result = generator(request.prompt, **all_gen_params)
                    logger.info(f"Generator returned: {type(result)}")
            else:
                # No lock, run directly (backward compatibility)
                logger.info("No lock, calling generator directly")
                result = generator(request.prompt, **all_gen_params)
            
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
            logger.error(f"Generation failed: {e}")
            import traceback
            tb = traceback.format_exc()
            logger.error(tb)
            # For debugging, include more info in the error
            error_info = {
                "error": str(e),
                "type": type(e).__name__,
                "traceback": tb.split('\n')[-5:]  # Last few lines
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