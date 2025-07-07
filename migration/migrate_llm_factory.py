#!/usr/bin/env python3
"""
Migration helper to convert LLMFactory usage to LLM Service.
This provides drop-in replacements for the old LLMFactory classes.
"""
import os
from typing import Optional, Literal, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.language_models import LLM
import json
import logging

logger = logging.getLogger(__name__)

# Default service configuration
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8000/v1")
LLM_SERVICE_API_KEY = os.getenv("LLM_SERVICE_API_KEY", "not-needed")


class LLMServiceAdapter:
    """Adapter that mimics the old LLMFactory interface using the new service."""
    
    # Map old profiles to new model tiers
    PROFILE_TO_TIER = {
        "classification": "light",
        "grading": "light", 
        "extraction": "medium",
        "generation": "medium",
        "rewriting": "medium",
    }
    
    def __init__(self, 
                 base_url: str = LLM_SERVICE_URL,
                 api_key: str = LLM_SERVICE_API_KEY):
        self.base_url = base_url
        self.api_key = api_key
        self._models_cache = {}
    
    def create_llm(self,
                   profile: Literal["classification", "grading", "extraction", "generation", "rewriting"],
                   model_override: Optional[str] = None,
                   use_cloud: bool = False,
                   use_grammar: bool = True,
                   use_structured_output: bool = False) -> LLM:
        """
        Create an LLM instance compatible with old LLMFactory interface.
        
        Args:
            profile: Task profile (maps to model tier)
            model_override: Override model selection
            use_cloud: Ignored (service handles this)
            use_grammar: Enable JSON grammar
            use_structured_output: Enable structured output
        
        Returns:
            LangChain LLM instance
        """
        # Determine model/tier
        if model_override:
            model = model_override
        else:
            model = self.PROFILE_TO_TIER.get(profile, "medium")
        
        # Cache key for reusing models
        cache_key = f"{model}_{profile}_{use_grammar}_{use_structured_output}"
        
        if cache_key in self._models_cache:
            return self._models_cache[cache_key]
        
        # Get task-specific parameters
        task_params = self._get_task_params(profile)
        
        # Create model kwargs
        model_kwargs = {}
        
        # Add grammar support
        if use_grammar or use_structured_output:
            grammar = self._get_grammar_for_profile(profile)
            if grammar:
                model_kwargs["grammar"] = grammar
            # Also set response format for JSON
            if profile in ["classification", "grading", "extraction"]:
                model_kwargs["response_format"] = {"type": "json_object"}
        
        # Create LLM instance
        llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=model,
            temperature=task_params["temperature"],
            max_tokens=task_params["max_tokens"],
            top_p=task_params.get("top_p", 0.95),
            model_kwargs=model_kwargs,
            timeout=task_params.get("timeout", 30.0),
        )
        
        # Cache for reuse
        self._models_cache[cache_key] = llm
        
        logger.info(f"Created LLM for {profile} using model {model}")
        return llm
    
    def _get_task_params(self, profile: str) -> Dict[str, Any]:
        """Get task-specific parameters matching old LLMFactory."""
        params = {
            "classification": {
                "temperature": 0.1,
                "max_tokens": 16,
                "top_p": 0.9,
                "timeout": 10.0,
            },
            "grading": {
                "temperature": 0.1,
                "max_tokens": 16,
                "top_p": 0.9,
                "timeout": 15.0,
            },
            "extraction": {
                "temperature": 0.1,
                "max_tokens": 2048,
                "top_p": 0.95,
                "timeout": 30.0,
            },
            "generation": {
                "temperature": 0.3,
                "max_tokens": 4096,
                "top_p": 0.95,
                "timeout": 45.0,
            },
            "rewriting": {
                "temperature": 0.5,
                "max_tokens": 2048,
                "top_p": 0.95,
                "timeout": 15.0,
            },
        }
        return params.get(profile, params["generation"])
    
    def _get_grammar_for_profile(self, profile: str) -> Optional[str]:
        """Get JSON grammar for specific profiles."""
        grammars = {
            "classification": '''root ::= object
object ::= "{" ws "\\"answer\\"" ws ":" ws answer ws "}"
answer ::= "\\"yes\\"" | "\\"no\\"" | "\\"relevant\\"" | "\\"not_relevant\\"" | "\\"true\\"" | "\\"false\\""
ws ::= [ \\t\\n]*''',
            
            "grading": '''root ::= object
object ::= "{" ws "\\"relevant\\"" ws ":" ws boolean ws "," ws "\\"score\\"" ws ":" ws number ws "}"
boolean ::= "true" | "false"
number ::= [0-9] | [0-9] "." [0-9]+
ws ::= [ \\t\\n]*''',
            
            "extraction": '''root ::= object
object ::= "{" ws "\\"entities\\"" ws ":" ws entity-array ws "," ws "\\"relations\\"" ws ":" ws relation-array ws "}"
entity-array ::= "[" ws "]" | "[" ws entity ("," ws entity)* ws "]"
relation-array ::= "[" ws "]" | "[" ws relation ("," ws relation)* ws "]"
entity ::= "{" ws entity-pairs ws "}"
relation ::= "{" ws relation-pairs ws "}"
entity-pairs ::= "\\"name\\"" ws ":" ws string ws "," ws "\\"type\\"" ws ":" ws entity-type ws "," ws "\\"context\\"" ws ":" ws string
relation-pairs ::= "\\"source\\"" ws ":" ws string ws "," ws "\\"target\\"" ws ":" ws string ws "," ws "\\"type\\"" ws ":" ws relation-type ws "," ws "\\"context\\"" ws ":" ws string
entity-type ::= "\\"case\\"" | "\\"statute\\"" | "\\"concept\\"" | "\\"court\\"" | "\\"rule\\"" | "\\"test\\""
relation-type ::= "\\"cites\\"" | "\\"establishes\\"" | "\\"interprets\\"" | "\\"applies\\"" | "\\"overrules\\"" | "\\"defines\\"" | "\\"requires\\""
string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""
ws ::= [ \\t\\n]*''',
        }
        return grammars.get(profile)
    
    def invoke_structured(self,
                         llm: LLM,
                         prompt: str,
                         expected_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke LLM and parse structured output (compatible with old interface).
        """
        try:
            # Invoke the LLM
            response = llm.invoke(prompt)
            
            # Handle response based on type
            if hasattr(response, 'content'):
                # AIMessage from ChatOpenAI
                response_text = response.content
            else:
                # String response
                response_text = str(response)
            
            # Try to parse JSON
            try:
                result = json.loads(response_text)
                
                # Validate against expected format if provided
                if expected_format:
                    for key in expected_format:
                        if key not in result:
                            logger.warning(f"Missing expected key: {key}")
                
                return result
                
            except json.JSONDecodeError as e:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        return result
                    except:
                        pass
                
                logger.error(f"JSON parse error: {e}")
                return {"error": "JSON parse error", "raw": response_text}
                
        except Exception as e:
            logger.error(f"LLM invocation error: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """Get cache statistics (queries the service)."""
        import requests
        try:
            response = requests.get(f"{LLM_SERVICE_URL}/internal/cache/stats")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {"error": "Cache stats not available"}


# Drop-in replacements for old imports
CachedLLMFactory = LLMServiceAdapter
LLMFactory = LLMServiceAdapter

# Singleton instance for compatibility
_default_factory = None

def get_llm_factory() -> LLMServiceAdapter:
    """Get singleton LLM factory instance."""
    global _default_factory
    if _default_factory is None:
        _default_factory = LLMServiceAdapter()
    return _default_factory


# Example migration function
def migrate_extraction_script():
    """Example of how to migrate an extraction script."""
    print("Migration Example:")
    print("-" * 60)
    print("OLD CODE:")
    print("""
from llm_factory_cached import CachedLLMFactory

# Create LLM
llm = CachedLLMFactory.create_llm(
    profile="extraction",
    use_grammar=True
)

# Use for extraction
response = llm.invoke(prompt)
    """)
    
    print("\nNEW CODE (Option 1 - Using compatibility layer):")
    print("""
from migrate_llm_factory import CachedLLMFactory

# Create LLM (same interface)
llm = CachedLLMFactory().create_llm(
    profile="extraction",
    use_grammar=True
)

# Use for extraction (same)
response = llm.invoke(prompt)
    """)
    
    print("\nNEW CODE (Option 2 - Direct service usage):")
    print("""
from langchain_openai import ChatOpenAI

# Create LLM
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="medium",
    temperature=0.1,
    max_tokens=2048,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# Use for extraction
response = llm.invoke(prompt)
    """)


if __name__ == "__main__":
    # Show migration example
    migrate_extraction_script()
    
    # Test compatibility layer
    print("\n\nTesting Compatibility Layer:")
    print("-" * 60)
    
    factory = LLMServiceAdapter()
    
    # Test creating different profile LLMs
    for profile in ["classification", "extraction", "generation"]:
        try:
            llm = factory.create_llm(profile)
            print(f"✓ Created {profile} LLM successfully")
        except Exception as e:
            print(f"✗ Failed to create {profile} LLM: {e}")