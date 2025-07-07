# Legal Research LLM Migration Guide

Quick reference for migrating Legal Research pipelines to use the new LLM Service.

## Quick Start

1. **Start the LLM Service**
   ```bash
   cd /Users/patrickpark/Documents/Work/utils/llmservice
   ./start_service.sh
   ```

2. **Update Environment Variables**
   ```bash
   # Add to scripts/datastore/.env
   LLM_SERVICE_URL=http://localhost:8000/v1
   LLM_SERVICE_API_KEY=not-needed
   ```

3. **Use Compatibility Layer** (Minimal changes)
   ```python
   # Old import
   # from llm_factory_cached import CachedLLMFactory
   
   # New import
   from migrate_llm_factory import CachedLLMFactory
   
   # Rest of code stays the same
   llm = CachedLLMFactory().create_llm(profile="extraction")
   ```

## Model Tier Mapping

| Old Profile | New Tier | Use Case |
|------------|----------|----------|
| classification | light | Query classification, yes/no |
| grading | light | Relevance scoring |
| extraction | medium | Entity extraction |
| generation | medium/heavy | Answer generation |
| rewriting | medium | Query enhancement |

## Common Migration Patterns

### Entity Extraction
```python
# Old
from llm_factory_cached import CachedLLMFactory
llm = CachedLLMFactory.create_llm("extraction", use_grammar=True)
result = CachedLLMFactory.invoke_structured(llm, prompt)

# New (Direct)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="medium",
    temperature=0.1,
    model_kwargs={"response_format": {"type": "json_object"}}
)
response = llm.invoke(prompt)
result = json.loads(response.content)
```

### Query Classification
```python
# Old
llm = CachedLLMFactory.create_llm("classification")

# New
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed", 
    model="light",
    temperature=0.1,
    max_tokens=20
)
```

### Relationship Extraction
```python
# Old
llm = CachedLLMFactory.create_llm("generation", model_override="mistral-24b")

# New
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="heavy",  # Heavy tier for complex tasks
    temperature=0.3,
    max_tokens=3000
)
```

## Batch Processing

```python
import asyncio
from langchain_openai import ChatOpenAI

async def batch_process(documents, max_concurrent=10):
    llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
        model="medium"
    )
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_one(doc):
        async with semaphore:
            return await llm.ainvoke(doc)
    
    results = await asyncio.gather(*[process_one(doc) for doc in documents])
    return results
```

## JSON Grammar Support

The new service supports JSON grammars through the `grammar` parameter:

```python
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="medium",
    model_kwargs={
        "response_format": {"type": "json_object"},
        "grammar": '''your_json_grammar_here'''
    }
)
```

## Performance Tips

1. **Use Model Tiers Appropriately**
   - Light: Fast classification tasks
   - Medium: Balanced extraction/generation
   - Heavy: Complex analysis

2. **Enable Caching** (in LLM Service .env)
   ```env
   ENABLE_CACHE=true
   CACHE_TTL_SECONDS=3600
   ```

3. **Batch Requests**
   - Use async methods for concurrent processing
   - Set appropriate `max_concurrent` limits

4. **Monitor Service Health**
   ```python
   import requests
   health = requests.get("http://localhost:8000/health").json()
   ```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Start LLM service: `./start_service.sh` |
| Model not found | Check model paths in `.env` |
| Slow responses | Use lighter model tier or enable GPU |
| JSON parse errors | Use `response_format` and grammar |

## Script-Specific Migration

### extract_entities_structured.py
- Change: Use `model="medium"` instead of `gemma-4b`
- Keep: JSON grammar for structured output

### extract_relationships_structured.py  
- Change: Use `model="heavy"` instead of `mistral-24b`
- Keep: Structured output format

### research_agent.py
- Change: Replace `CachedLLMFactory` with service calls
- Keep: Multi-tier model selection logic

### hyde_enhancer.py
- Change: Use `model="light"` for query classification
- Keep: Query enhancement logic