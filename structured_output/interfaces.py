"""
Abstract interfaces for structured output generation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_manager import ModelTier

@dataclass
class StructuredRequest:
    """Request for structured generation."""
    prompt: str
    response_format: Dict[str, Any]
    model_config: Dict[str, Any]
    generation_params: Dict[str, Any]
    model_tier: 'ModelTier'


@dataclass
class StructuredResponse:
    """Response from structured generation."""
    content: str
    parsed_result: Any
    processing_time_ms: float
    backend_used: str
    refusal: Optional[str] = None  # For OpenAI compatibility


class StructuredBackend(ABC):
    """Abstract interface for structured generation backends."""
    
    @abstractmethod
    async def generate(self, request: StructuredRequest) -> StructuredResponse:
        """Generate structured output according to request."""
        pass
    
    @abstractmethod
    def supports_format_type(self, format_type: str) -> bool:
        """Check if backend supports given format type."""
        pass
    
    @abstractmethod
    def compile_schema(self, response_format: Dict[str, Any]) -> str:
        """
        Convert schema to backend-specific format.
        Returns a cache key for the compiled schema.
        """
        pass