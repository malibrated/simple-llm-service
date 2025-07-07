# Legal Research Integration Guide

This guide shows how to integrate the Legal Research ingestion and retrieval pipelines with the new LLM Service.

## Overview

The Legal Research system uses LLMs for:
1. **Entity Extraction** - Identifying legal entities (cases, statutes, courts, etc.)
2. **Relationship Extraction** - Finding relationships between entities
3. **Query Analysis** - Classifying and enhancing search queries
4. **Answer Generation** - Creating comprehensive responses with citations

## Configuration

### 1. Update Legal Research .env

Add the LLM service URL to your Legal Research environment:

```bash
# scripts/datastore/.env
LLM_SERVICE_URL=http://localhost:8000/v1
LLM_SERVICE_API_KEY=not-needed  # Local service doesn't need API key
```

### 2. Configure LLM Service Models

Set up models in the LLM service `.env` for legal research tasks:

```bash
# /Users/patrickpark/Documents/Work/utils/llmservice/.env

# LIGHT - Query classification and grading
LIGHT_MODEL_PATH=/Users/patrickpark/.cache/lm-studio/models/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0.gguf
LIGHT_BACKEND=llamacpp
LIGHT_TEMPERATURE=0.1
LIGHT_MAX_TOKENS=100
LIGHT_TOP_K=10

# MEDIUM - Entity extraction and answer generation
MEDIUM_MODEL_PATH=/Users/patrickpark/.cache/lm-studio/models/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_S.gguf
MEDIUM_BACKEND=llamacpp
MEDIUM_TEMPERATURE=0.3
MEDIUM_MAX_TOKENS=2048
MEDIUM_N_CTX=131072  # Large context for legal documents

# HEAVY - Relationship extraction and complex analysis
HEAVY_MODEL_PATH=/Users/patrickpark/.cache/lm-studio/models/Mistral-Small-3.2-24B-Instruct-GGUF/Mistral-Small-3.2-24B-Instruct-Q4_K_M.gguf
HEAVY_BACKEND=llamacpp
HEAVY_TEMPERATURE=0.5
HEAVY_MAX_TOKENS=4096
HEAVY_N_CTX=131072
```

## Integration Examples

### 1. Entity Extraction Pipeline

Create a new entity extraction script using the LLM service:

```python
# scripts/datastore/extract_entities_service.py
import os
import json
import asyncio
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from database import get_session, Document, Entity, EntityMention

# Entity extraction prompt template
ENTITY_PROMPT = """Extract legal entities from the following text. Return a JSON object with an "entities" array.

Each entity should have:
- name: The entity name (e.g., "Smith v. Jones", "28 U.S.C. § 1331")
- type: One of: case, statute, court, concept, rule, test
- context: Brief context where the entity appears

Text to analyze:
{text}

Return JSON in this exact format:
{{
  "entities": [
    {{"name": "...", "type": "...", "context": "..."}}
  ]
}}"""

class EntityExtractorService:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=os.getenv("LLM_SERVICE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("LLM_SERVICE_API_KEY", "not-needed"),
            model="medium",
            temperature=0.1,
            model_kwargs={
                "response_format": {"type": "json_object"},
                # Can also specify grammar for more control
                "grammar": '''
root ::= object
object ::= "{" ws "\\"entities\\"" ws ":" ws array ws "}"
array ::= "[" ws "]" | "[" ws entity ("," ws entity)* ws "]"
entity ::= "{" ws "\\"name\\"" ws ":" ws string ws "," ws "\\"type\\"" ws ":" ws type ws "," ws "\\"context\\"" ws ":" ws string ws "}"
type ::= "\\"case\\"" | "\\"statute\\"" | "\\"court\\"" | "\\"concept\\"" | "\\"rule\\"" | "\\"test\\""
string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""
ws ::= [ \\t\\n]*'''
            }
        )
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using LLM service."""
        try:
            prompt = ENTITY_PROMPT.format(text=text[:4000])  # Limit text length
            response = await self.llm.ainvoke(prompt)
            
            # Parse JSON response
            result = json.loads(response.content)
            return result.get("entities", [])
            
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []
    
    async def process_document(self, doc_id: int):
        """Process a single document for entity extraction."""
        with get_session() as session:
            doc = session.query(Document).filter_by(id=doc_id).first()
            if not doc:
                return
            
            # Extract entities from content
            entities = await self.extract_entities(doc.content)
            
            # Store in database
            for entity_data in entities:
                # Create or get entity
                entity = session.query(Entity).filter_by(
                    name=entity_data["name"],
                    entity_type=entity_data["type"]
                ).first()
                
                if not entity:
                    entity = Entity(
                        name=entity_data["name"],
                        entity_type=entity_data["type"]
                    )
                    session.add(entity)
                    session.flush()
                
                # Create mention
                mention = EntityMention(
                    entity_id=entity.id,
                    document_id=doc.id,
                    context=entity_data["context"],
                    confidence=0.9  # High confidence for LLM extraction
                )
                session.add(mention)
            
            session.commit()

# Usage
async def main():
    extractor = EntityExtractorService()
    await extractor.process_document(123)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Relationship Extraction Pipeline

```python
# scripts/datastore/extract_relationships_service.py
import os
import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from database import get_session, Entity, Relationship

RELATIONSHIP_PROMPT = """Analyze the following legal text and identify relationships between the entities provided.

Entities found in this text:
{entities}

Text:
{text}

For each relationship found, provide:
- source: The source entity name
- target: The target entity name  
- type: One of: cites, establishes, interprets, applies, overrules, defines, requires
- context: Brief explanation of the relationship

Return JSON:
{{
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "context": "..."}}
  ]
}}"""

class RelationshipExtractorService:
    def __init__(self):
        # Use heavy model for complex relationship understanding
        self.llm = ChatOpenAI(
            base_url=os.getenv("LLM_SERVICE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("LLM_SERVICE_API_KEY", "not-needed"),
            model="heavy",  # Use heavy tier for relationships
            temperature=0.3,
            max_tokens=2048,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    
    async def extract_relationships(self, text: str, entities: List[str]) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        if len(entities) < 2:
            return []
            
        entities_str = "\n".join(f"- {e}" for e in entities)
        prompt = RELATIONSHIP_PROMPT.format(
            entities=entities_str,
            text=text[:6000]  # Larger context for relationships
        )
        
        try:
            response = await self.llm.ainvoke(prompt)
            result = json.loads(response.content)
            return result.get("relationships", [])
        except Exception as e:
            print(f"Error extracting relationships: {e}")
            return []
```

### 3. Research Agent Integration

Update the MCP research agent to use the service:

```python
# scripts/legal-research-mcp/tools/research_service.py
import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain_core.messages import SystemMessage, HumanMessage

class LegalResearchAgent:
    def __init__(self):
        # Configure different models for different tasks
        self.classifier_llm = ChatOpenAI(
            base_url=os.getenv("LLM_SERVICE_URL", "http://localhost:8000/v1"),
            api_key="not-needed",
            model="light",  # Fast classification
            temperature=0.1,
            max_tokens=20
        )
        
        self.analyzer_llm = ChatOpenAI(
            base_url=os.getenv("LLM_SERVICE_URL", "http://localhost:8000/v1"),
            api_key="not-needed",
            model="medium",  # Balanced analysis
            temperature=0.3,
            max_tokens=2048
        )
        
        self.generator_llm = ChatOpenAI(
            base_url=os.getenv("LLM_SERVICE_URL", "http://localhost:8000/v1"),
            api_key="not-needed",
            model="heavy",  # Best quality for final answer
            temperature=0.5,
            max_tokens=4096
        )
    
    async def classify_query(self, query: str) -> str:
        """Classify query complexity using light model."""
        prompt = f"Classify this legal query as 'simple' or 'complex': {query}"
        response = await self.classifier_llm.ainvoke(prompt)
        return response.content.strip().lower()
    
    async def analyze_documents(self, query: str, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze retrieved documents."""
        context = "\n\n".join([
            f"Document {i+1}: {doc['content'][:500]}..."
            for i, doc in enumerate(documents[:5])
        ])
        
        prompt = f"""Analyze these legal documents in relation to the query.
        
Query: {query}

Documents:
{context}

Provide a structured analysis including key findings and relevant citations."""
        
        response = await self.analyzer_llm.ainvoke(prompt)
        return {"analysis": response.content}
    
    async def generate_answer(self, query: str, analysis: Dict[str, Any], entities: List[str]) -> str:
        """Generate comprehensive answer using heavy model."""
        prompt = f"""Based on the analysis, provide a comprehensive answer to this legal question.

Question: {query}

Analysis: {analysis['analysis']}

Key Legal Entities: {', '.join(entities)}

Provide a detailed answer with proper legal citations and reasoning."""
        
        response = await self.generator_llm.ainvoke(prompt)
        return response.content
```

### 4. Batch Processing with Service

```python
# scripts/datastore/batch_process_service.py
import asyncio
import aiohttp
from typing import List, Dict, Any
import json

class BatchProcessor:
    def __init__(self, service_url: str = "http://localhost:8000/v1"):
        self.service_url = service_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def process_batch(self, texts: List[str], model: str = "medium", 
                          max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Process multiple texts concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_one(text: str) -> Dict[str, Any]:
            async with semaphore:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": f"Extract entities from: {text[:2000]}"}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"}
                }
                
                async with self.session.post(
                    f"{self.service_url}/chat/completions",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        return json.loads(content)
                    else:
                        print(f"Error: {response.status}")
                        return {"entities": []}
        
        # Process all texts concurrently
        results = await asyncio.gather(*[process_one(text) for text in texts])
        return results

# Usage
async def main():
    documents = ["Legal text 1...", "Legal text 2...", "Legal text 3..."]
    
    async with BatchProcessor() as processor:
        results = await processor.process_batch(documents, model="medium")
        for i, result in enumerate(results):
            print(f"Document {i+1}: {len(result.get('entities', []))} entities found")
```

## Migration Steps

### 1. Start LLM Service

```bash
cd /Users/patrickpark/Documents/Work/utils/llmservice
./start_service.sh
```

### 2. Update Imports

Replace current LLM imports:

```python
# Old
from llm_factory_cached import CachedLLMFactory

# New
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="medium"
)
```

### 3. Update Configuration

Add service configuration to your scripts:

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

LLM_CONFIG = {
    "base_url": os.getenv("LLM_SERVICE_URL", "http://localhost:8000/v1"),
    "api_key": os.getenv("LLM_SERVICE_API_KEY", "not-needed"),
    "models": {
        "classifier": "light",
        "extractor": "medium",
        "analyzer": "heavy"
    }
}
```

### 4. Update Extraction Scripts

Modify existing extraction scripts to use the service:

```python
# Simple wrapper for backwards compatibility
class LLMServiceWrapper:
    def __init__(self, tier: str = "medium"):
        self.llm = ChatOpenAI(
            base_url=LLM_CONFIG["base_url"],
            api_key=LLM_CONFIG["api_key"],
            model=tier
        )
    
    def invoke(self, prompt: str, **kwargs):
        """Synchronous invoke for compatibility."""
        return self.llm.invoke(prompt, **kwargs).content
    
    async def ainvoke(self, prompt: str, **kwargs):
        """Async invoke."""
        response = await self.llm.ainvoke(prompt, **kwargs)
        return response.content

# Drop-in replacement
def create_llm(profile: str = "extraction"):
    tier_map = {
        "classification": "light",
        "extraction": "medium",
        "generation": "heavy"
    }
    return LLMServiceWrapper(tier_map.get(profile, "medium"))
```

## Performance Optimization

### 1. Enable Caching

The LLM service includes built-in caching:

```bash
# .env
ENABLE_CACHE=true
CACHE_TTL_SECONDS=3600  # 1 hour cache for extraction results
CACHE_PERSIST_TO_DISK=true  # Persist across restarts
```

### 2. Batch Processing

Use concurrent requests for better throughput:

```python
async def batch_extract_entities(documents: List[Document], max_concurrent: int = 10):
    """Extract entities from multiple documents concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def extract_one(doc):
        async with semaphore:
            return await extractor.process_document(doc.id)
    
    await asyncio.gather(*[extract_one(doc) for doc in documents])
```

### 3. Model Selection

Choose appropriate models for each task:

- **Light tier**: Query classification, simple yes/no decisions
- **Medium tier**: Entity extraction, document analysis
- **Heavy tier**: Complex relationship extraction, comprehensive answers

## Monitoring

Track service performance:

```python
import httpx

# Check service health
response = httpx.get("http://localhost:8000/health")

# Get model list
response = httpx.get("http://localhost:8000/v1/models")

# Monitor with logs
# tail -f logs/llm_service.log
```

## Benefits

1. **Centralized Model Management**: All models configured in one place
2. **Better Resource Utilization**: Models stay loaded, auto-shutdown when idle
3. **Consistent API**: OpenAI-compatible interface across all pipelines
4. **Built-in Caching**: Reduces redundant processing
5. **Concurrent Processing**: Better throughput with async support
6. **Flexible Deployment**: Can run on separate machine or container