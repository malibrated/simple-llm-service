#!/usr/bin/env python3
"""Test script to verify lazy loading behavior."""
import asyncio
import logging
import os
from pathlib import Path
from model_manager import ModelManager, ModelTier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_lazy_loading():
    """Test that models are loaded only when needed."""
    # Create model manager
    manager = ModelManager()
    
    # Initialize (should only load configs, not models)
    logger.info("Initializing model manager...")
    await manager.initialize()
    
    # Check that no models are loaded yet
    logger.info(f"Models loaded after init: {list(manager.models.keys())}")
    logger.info(f"Configs available: {list(manager.configs.keys())}")
    
    # List available models (should show configured but unloaded)
    available = manager.get_available_models()
    logger.info(f"Available models: {available}")
    
    # Try to use a model (should trigger lazy loading)
    try:
        # Find first configured tier
        for tier in ModelTier:
            if tier in manager.configs:
                logger.info(f"\nTrying to generate with {tier.value} model...")
                result = await manager.generate(
                    model_tier=tier,
                    prompt="Hello",
                    max_tokens=5
                )
                logger.info(f"Generation successful: {result['text'][:50]}...")
                logger.info(f"Models loaded after generation: {list(manager.models.keys())}")
                break
    except Exception as e:
        logger.error(f"Generation failed: {e}")
    
    # List models again (should show loaded status)
    available = manager.get_available_models()
    logger.info(f"\nAvailable models after use: {available}")
    
    # Cleanup
    await manager.cleanup()

if __name__ == "__main__":
    # Need at least one model configured
    if not any(os.getenv(f"{tier.value.upper()}_MODEL_PATH") for tier in ModelTier):
        logger.warning("No models configured. Set LIGHT_MODEL_PATH or similar in .env")
        logger.info("Example: LIGHT_MODEL_PATH=/path/to/model.gguf")
    else:
        asyncio.run(test_lazy_loading())