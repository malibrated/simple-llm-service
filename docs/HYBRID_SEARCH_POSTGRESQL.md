# Hybrid Vector Search with PostgreSQL and pgvector

This guide covers implementing hybrid dense and sparse vector search using PostgreSQL with the pgvector extension, optimized for BGE-M3 and similar models that support both embedding types.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Schema Design](#schema-design)
- [Implementation](#implementation)
- [Performance Optimization](#performance-optimization)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

## Overview

### Why Hybrid Search?
- **Dense embeddings**: Capture semantic meaning (good for concepts, synonyms)
- **Sparse embeddings**: Capture exact terms (good for specific keywords, names)
- **Hybrid**: Best of both worlds - semantic understanding + precision

### Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   BGE-M3    │────▶│  LLM Service │────▶│   PostgreSQL    │
│   Encoder   │     │  (Fast API)  │     │   + pgvector    │
└─────────────┘     └──────────────┘     └─────────────────┘
                           │                      │
                           ▼                      ▼
                    ┌──────────────┐     ┌────────────────┐
                    │Dense Vector  │     │ Sparse Vector  │
                    │  (pgvector)  │     │    (JSONB)     │
                    └──────────────┘     └────────────────┘
```

## Installation

### 1. Install PostgreSQL with pgvector

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-15-pgvector  # Adjust version as needed

# macOS with Homebrew
brew install postgresql@15
brew install pgvector

# From source (for latest version)
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install  # May need sudo
```

### 2. Enable pgvector Extension

```sql
-- Connect to your database
psql -U postgres -d your_database

-- Enable the extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### 3. Python Dependencies

```bash
pip install asyncpg psycopg2-binary pgvector numpy
```

## Schema Design

### Basic Schema

```sql
-- Main documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dense embeddings table (using pgvector)
CREATE TABLE dense_embeddings (
    document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    embedding vector(1024),  -- BGE-M3 outputs 1024 dimensions
    model_version VARCHAR(50) DEFAULT 'bge-m3-v1'
);

-- Sparse embeddings table (normalized format)
CREATE TABLE sparse_embeddings (
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    token_id INTEGER NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY (document_id, token_id)
);

-- Alternative: Single table with JSONB for sparse
CREATE TABLE embeddings (
    document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    dense_embedding vector(1024),
    sparse_embedding JSONB,  -- {"token_id": weight, ...}
    model_version VARCHAR(50) DEFAULT 'bge-m3-v1',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_dense_embeddings_vector ON dense_embeddings 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_sparse_embeddings_doc_id ON sparse_embeddings (document_id);
CREATE INDEX idx_sparse_embeddings_token_id ON sparse_embeddings (token_id);
CREATE INDEX idx_embeddings_sparse ON embeddings USING GIN (sparse_embedding);
```

### Optimized Schema for Hybrid Search

```sql
-- Materialized view for faster hybrid queries
CREATE MATERIALIZED VIEW document_search AS
SELECT 
    d.id,
    d.content,
    d.metadata,
    de.embedding as dense_embedding,
    COALESCE(
        json_object_agg(se.token_id::text, se.weight) 
        FILTER (WHERE se.token_id IS NOT NULL), 
        '{}'::json
    ) as sparse_embedding
FROM documents d
LEFT JOIN dense_embeddings de ON d.id = de.document_id
LEFT JOIN sparse_embeddings se ON d.id = se.document_id
GROUP BY d.id, d.content, d.metadata, de.embedding;

-- Refresh periodically
CREATE INDEX idx_document_search_dense ON document_search 
    USING ivfflat (dense_embedding vector_cosine_ops);
```

## Implementation

### Database Connection Pool

```python
import asyncpg
import numpy as np
from typing import List, Dict, Tuple, Optional
import json

class HybridVectorDB:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool = None
        
    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=10,
            max_size=20,
            command_timeout=60
        )
        
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
```

### Inserting Embeddings

```python
async def insert_document(
    self,
    content: str,
    dense_embedding: List[float],
    sparse_embedding: Dict[int, float],
    metadata: Optional[Dict] = None
) -> int:
    """Insert document with both dense and sparse embeddings"""
    async with self.pool.acquire() as conn:
        async with conn.transaction():
            # Insert document
            doc_id = await conn.fetchval("""
                INSERT INTO documents (content, metadata)
                VALUES ($1, $2::jsonb)
                RETURNING id
            """, content, json.dumps(metadata or {}))
            
            # Insert dense embedding
            await conn.execute("""
                INSERT INTO dense_embeddings (document_id, embedding)
                VALUES ($1, $2::vector)
            """, doc_id, dense_embedding)
            
            # Insert sparse embeddings (batch insert for efficiency)
            if sparse_embedding:
                sparse_records = [
                    (doc_id, int(token_id), float(weight))
                    for token_id, weight in sparse_embedding.items()
                ]
                await conn.executemany("""
                    INSERT INTO sparse_embeddings (document_id, token_id, weight)
                    VALUES ($1, $2, $3)
                """, sparse_records)
                
            return doc_id

async def batch_insert_documents(
    self,
    documents: List[Dict]
) -> List[int]:
    """Batch insert multiple documents"""
    async with self.pool.acquire() as conn:
        # Use COPY for best performance
        async with conn.transaction():
            # First, insert all documents
            doc_ids = await conn.fetch("""
                INSERT INTO documents (content, metadata)
                SELECT content, metadata::jsonb
                FROM unnest($1::text[], $2::jsonb[]) AS t(content, metadata)
                RETURNING id
            """, 
            [d['content'] for d in documents],
            [json.dumps(d.get('metadata', {})) for d in documents]
            )
            
            # Prepare dense embeddings for COPY
            dense_data = []
            for doc_id, doc in zip(doc_ids, documents):
                dense_data.append((
                    doc_id['id'],
                    doc['dense_embedding']
                ))
            
            # Bulk insert dense embeddings
            await conn.copy_records_to_table(
                'dense_embeddings',
                records=dense_data,
                columns=['document_id', 'embedding']
            )
            
            # Bulk insert sparse embeddings
            sparse_data = []
            for doc_id, doc in zip(doc_ids, documents):
                for token_id, weight in doc['sparse_embedding'].items():
                    sparse_data.append((
                        doc_id['id'],
                        int(token_id),
                        float(weight)
                    ))
                    
            if sparse_data:
                await conn.copy_records_to_table(
                    'sparse_embeddings',
                    records=sparse_data,
                    columns=['document_id', 'token_id', 'weight']
                )
                
            return [d['id'] for d in doc_ids]
```

### Search Implementation

```python
async def dense_search(
    self,
    query_embedding: List[float],
    limit: int = 10,
    threshold: float = 0.0
) -> List[Tuple[int, float, str]]:
    """Perform dense vector similarity search"""
    async with self.pool.acquire() as conn:
        results = await conn.fetch("""
            SELECT 
                d.id,
                d.content,
                1 - (de.embedding <=> $1::vector) as similarity
            FROM documents d
            JOIN dense_embeddings de ON d.id = de.document_id
            WHERE 1 - (de.embedding <=> $1::vector) > $2
            ORDER BY de.embedding <=> $1::vector
            LIMIT $3
        """, query_embedding, threshold, limit)
        
        return [(r['id'], r['similarity'], r['content']) for r in results]

async def sparse_search(
    self,
    query_sparse: Dict[int, float],
    limit: int = 10,
    threshold: float = 0.0
) -> List[Tuple[int, float, str]]:
    """Perform sparse vector similarity search"""
    if not query_sparse:
        return []
        
    # Convert sparse dict to arrays for SQL
    token_ids = list(query_sparse.keys())
    weights = list(query_sparse.values())
    
    async with self.pool.acquire() as conn:
        results = await conn.fetch("""
            WITH query_tokens AS (
                SELECT 
                    unnest($1::integer[]) as token_id,
                    unnest($2::real[]) as weight
            ),
            scores AS (
                SELECT 
                    se.document_id,
                    SUM(se.weight * qt.weight) as score
                FROM sparse_embeddings se
                JOIN query_tokens qt ON se.token_id = qt.token_id
                GROUP BY se.document_id
                HAVING SUM(se.weight * qt.weight) > $3
            )
            SELECT 
                d.id,
                d.content,
                s.score
            FROM scores s
            JOIN documents d ON d.id = s.document_id
            ORDER BY s.score DESC
            LIMIT $4
        """, token_ids, weights, threshold, limit)
        
        return [(r['id'], r['score'], r['content']) for r in results]

async def hybrid_search(
    self,
    query_dense: List[float],
    query_sparse: Dict[int, float],
    limit: int = 10,
    alpha: float = 0.5,  # Weight for dense vs sparse (0=sparse only, 1=dense only)
    rerank_top_k: int = 100
) -> List[Tuple[int, float, str, Dict]]:
    """
    Perform hybrid search combining dense and sparse results
    Uses Reciprocal Rank Fusion (RRF) for combination
    """
    # Get top-k from each method
    dense_results = await self.dense_search(query_dense, rerank_top_k)
    sparse_results = await self.sparse_search(query_sparse, rerank_top_k)
    
    # Create score dictionaries
    dense_scores = {doc_id: score for doc_id, score, _ in dense_results}
    sparse_scores = {doc_id: score for doc_id, score, _ in sparse_results}
    
    # Get all unique document IDs
    all_doc_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
    
    # Reciprocal Rank Fusion
    k = 60  # RRF parameter
    doc_contents = {}
    rrf_scores = {}
    
    # Calculate RRF scores
    for i, (doc_id, _, content) in enumerate(dense_results):
        doc_contents[doc_id] = content
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + alpha / (k + i + 1)
        
    for i, (doc_id, _, content) in enumerate(sparse_results):
        doc_contents[doc_id] = content
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - alpha) / (k + i + 1)
    
    # Sort by combined score
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:limit]
    
    # Fetch full results with metadata
    doc_ids = [doc_id for doc_id, _ in sorted_results]
    async with self.pool.acquire() as conn:
        docs = await conn.fetch("""
            SELECT id, content, metadata
            FROM documents
            WHERE id = ANY($1::integer[])
        """, doc_ids)
        
    doc_map = {d['id']: d for d in docs}
    
    # Format results
    final_results = []
    for doc_id, score in sorted_results:
        doc = doc_map.get(doc_id)
        if doc:
            final_results.append((
                doc_id,
                score,
                doc['content'],
                {
                    'metadata': doc['metadata'],
                    'dense_score': dense_scores.get(doc_id, 0),
                    'sparse_score': sparse_scores.get(doc_id, 0)
                }
            ))
            
    return final_results
```

### Advanced Hybrid Search with Normalization

```python
async def hybrid_search_normalized(
    self,
    query_dense: List[float],
    query_sparse: Dict[int, float],
    limit: int = 10,
    alpha: float = 0.5,
    normalize_scores: bool = True
) -> List[Tuple[int, float, str, Dict]]:
    """
    Hybrid search with score normalization
    """
    async with self.pool.acquire() as conn:
        # Combined query with CTEs
        results = await conn.fetch("""
            WITH 
            -- Dense search with similarity scores
            dense_scores AS (
                SELECT 
                    de.document_id,
                    1 - (de.embedding <=> $1::vector) as score,
                    ROW_NUMBER() OVER (ORDER BY de.embedding <=> $1::vector) as rank
                FROM dense_embeddings de
                ORDER BY de.embedding <=> $1::vector
                LIMIT $2
            ),
            -- Sparse search with dot product scores
            sparse_scores AS (
                WITH query_tokens AS (
                    SELECT 
                        unnest($3::integer[]) as token_id,
                        unnest($4::real[]) as weight
                )
                SELECT 
                    se.document_id,
                    SUM(se.weight * qt.weight) as score,
                    ROW_NUMBER() OVER (ORDER BY SUM(se.weight * qt.weight) DESC) as rank
                FROM sparse_embeddings se
                JOIN query_tokens qt ON se.token_id = qt.token_id
                GROUP BY se.document_id
                ORDER BY score DESC
                LIMIT $2
            ),
            -- Normalize scores if requested
            normalized_scores AS (
                SELECT 
                    COALESCE(d.document_id, s.document_id) as document_id,
                    CASE 
                        WHEN $5 THEN 
                            -- Min-max normalization
                            COALESCE(
                                (d.score - MIN(d.score) OVER()) / 
                                NULLIF(MAX(d.score) OVER() - MIN(d.score) OVER(), 0), 
                                0
                            )
                        ELSE d.score 
                    END as dense_norm,
                    CASE 
                        WHEN $5 THEN 
                            COALESCE(
                                (s.score - MIN(s.score) OVER()) / 
                                NULLIF(MAX(s.score) OVER() - MIN(s.score) OVER(), 0), 
                                0
                            )
                        ELSE s.score 
                    END as sparse_norm,
                    d.score as dense_raw,
                    s.score as sparse_raw,
                    d.rank as dense_rank,
                    s.rank as sparse_rank
                FROM dense_scores d
                FULL OUTER JOIN sparse_scores s ON d.document_id = s.document_id
            )
            -- Final scoring and retrieval
            SELECT 
                doc.id,
                doc.content,
                doc.metadata,
                -- Weighted combination
                ($6 * COALESCE(ns.dense_norm, 0) + 
                 (1 - $6) * COALESCE(ns.sparse_norm, 0)) as final_score,
                ns.dense_raw,
                ns.sparse_raw,
                ns.dense_rank,
                ns.sparse_rank
            FROM normalized_scores ns
            JOIN documents doc ON doc.id = ns.document_id
            ORDER BY final_score DESC
            LIMIT $7
        """, 
        query_dense,      # $1
        rerank_top_k,     # $2
        list(query_sparse.keys()),    # $3
        list(query_sparse.values()),  # $4
        normalize_scores, # $5
        alpha,           # $6
        limit            # $7
        )
        
        return [
            (
                r['id'],
                r['final_score'],
                r['content'],
                {
                    'metadata': r['metadata'],
                    'dense_score': r['dense_raw'],
                    'sparse_score': r['sparse_raw'],
                    'dense_rank': r['dense_rank'],
                    'sparse_rank': r['sparse_rank']
                }
            )
            for r in results
        ]
```

## Performance Optimization

### 1. Indexing Strategies

```sql
-- For dense vectors (choose one based on your needs)

-- IVFFlat: Faster build, good recall, less memory
CREATE INDEX idx_dense_ivfflat ON dense_embeddings 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);  -- Adjust lists based on dataset size

-- HNSW: Slower build, better recall, more memory (pgvector 0.5.0+)
CREATE INDEX idx_dense_hnsw ON dense_embeddings 
    USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);

-- For sparse vectors
CREATE INDEX idx_sparse_token ON sparse_embeddings (token_id, document_id);
CREATE INDEX idx_sparse_weight ON sparse_embeddings (weight) 
    WHERE weight > 0.1;  -- Only index significant weights

-- Composite indexes for common queries
CREATE INDEX idx_documents_created ON documents (created_at DESC);
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);
```

### 2. Query Optimization

```python
# Prefiltering for better performance
async def hybrid_search_with_filter(
    self,
    query_dense: List[float],
    query_sparse: Dict[int, float],
    filter_condition: str,
    filter_params: List,
    limit: int = 10
) -> List[Tuple[int, float, str]]:
    """Hybrid search with metadata filtering"""
    async with self.pool.acquire() as conn:
        # Create temporary table with filtered IDs
        await conn.execute(f"""
            CREATE TEMP TABLE filtered_docs AS
            SELECT id FROM documents
            WHERE {filter_condition}
        """, *filter_params)
        
        # Now perform hybrid search only on filtered documents
        # ... (similar to above but with JOIN on filtered_docs)
```

### 3. Caching Strategies

```python
import hashlib
from functools import lru_cache

class CachedHybridSearch(HybridVectorDB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
        
    def _get_cache_key(
        self, 
        query_dense: List[float], 
        query_sparse: Dict[int, float],
        **kwargs
    ) -> str:
        """Generate cache key for query"""
        dense_hash = hashlib.md5(
            np.array(query_dense).tobytes()
        ).hexdigest()[:8]
        
        sparse_hash = hashlib.md5(
            json.dumps(sorted(query_sparse.items())).encode()
        ).hexdigest()[:8]
        
        return f"{dense_hash}_{sparse_hash}_{kwargs}"
    
    async def hybrid_search_cached(self, *args, **kwargs):
        cache_key = self._get_cache_key(*args[:2], **kwargs)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        results = await self.hybrid_search(*args, **kwargs)
        self._cache[cache_key] = results
        
        # Limit cache size
        if len(self._cache) > 1000:
            self._cache.pop(next(iter(self._cache)))
            
        return results
```

### 4. Connection Pool Tuning

```python
# Optimized connection pool settings
async def create_optimized_pool(dsn: str):
    return await asyncpg.create_pool(
        dsn,
        min_size=5,
        max_size=20,
        max_queries=50000,
        max_cached_statement_lifetime=600,
        max_inactive_connection_lifetime=60,
        command_timeout=30,
        statement_cache_size=2000,
        # For read-heavy workloads
        server_settings={
            'jit': 'off',  # Disable JIT for consistent performance
            'random_page_cost': '1.1',  # For SSD storage
            'effective_cache_size': '4GB',
            'shared_buffers': '1GB',
            'work_mem': '50MB'
        }
    )
```

## Usage Examples

### Complete Integration Example

```python
# integration_example.py
import asyncio
import aiohttp
from typing import List, Dict, Tuple

class LLMServiceHybridSearch:
    def __init__(self, llm_service_url: str, db_dsn: str):
        self.llm_service_url = llm_service_url
        self.db = HybridVectorDB(db_dsn)
        
    async def initialize(self):
        await self.db.initialize()
        
    async def index_document(self, content: str, metadata: Dict = None):
        """Index a document using LLM service for embeddings"""
        async with aiohttp.ClientSession() as session:
            # Get embeddings from LLM service
            async with session.post(
                f"{self.llm_service_url}/v1/embeddings",
                json={
                    "model": "embedding",
                    "input": content,
                    "embedding_type": "dense",
                    "return_sparse": True
                }
            ) as resp:
                result = await resp.json()
                
            embedding_data = result['data'][0]
            dense_embedding = embedding_data['embedding']
            sparse_embedding = embedding_data.get('sparse_embedding', {})
            
            # Store in database
            doc_id = await self.db.insert_document(
                content=content,
                dense_embedding=dense_embedding,
                sparse_embedding=sparse_embedding,
                metadata=metadata
            )
            
            return doc_id
    
    async def search(
        self, 
        query: str, 
        limit: int = 10,
        alpha: float = 0.5
    ) -> List[Dict]:
        """Search using hybrid approach"""
        async with aiohttp.ClientSession() as session:
            # Get query embeddings
            async with session.post(
                f"{self.llm_service_url}/v1/embeddings",
                json={
                    "model": "embedding",
                    "input": query,
                    "embedding_type": "dense",
                    "return_sparse": True
                }
            ) as resp:
                result = await resp.json()
                
            embedding_data = result['data'][0]
            query_dense = embedding_data['embedding']
            query_sparse = embedding_data.get('sparse_embedding', {})
            
            # Perform hybrid search
            results = await self.db.hybrid_search(
                query_dense=query_dense,
                query_sparse=query_sparse,
                limit=limit,
                alpha=alpha
            )
            
            # Format results
            formatted_results = []
            for doc_id, score, content, meta in results:
                formatted_results.append({
                    'id': doc_id,
                    'score': score,
                    'content': content,
                    'metadata': meta['metadata'],
                    'debug': {
                        'dense_score': meta['dense_score'],
                        'sparse_score': meta['sparse_score']
                    }
                })
                
            return formatted_results

# Usage
async def main():
    service = LLMServiceHybridSearch(
        llm_service_url="http://localhost:8000",
        db_dsn="postgresql://user:pass@localhost/vectordb"
    )
    
    await service.initialize()
    
    # Index documents
    doc_id = await service.index_document(
        "PostgreSQL is a powerful open-source relational database",
        metadata={"category": "database", "source": "docs"}
    )
    
    # Search
    results = await service.search(
        "open source database with vector support",
        limit=5,
        alpha=0.7  # Favor dense embeddings
    )
    
    for result in results:
        print(f"Score: {result['score']:.3f} - {result['content'][:50]}...")
        print(f"  Dense: {result['debug']['dense_score']:.3f}, "
              f"Sparse: {result['debug']['sparse_score']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

### 1. **Embedding Storage**
- Store embeddings separately from documents for flexibility
- Use appropriate precision (REAL vs DOUBLE PRECISION)
- Consider compression for large-scale deployments

### 2. **Sparse Embedding Optimization**
- Only store non-zero weights
- Consider threshold filtering (e.g., weight > 0.01)
- Use batch operations for insertion

### 3. **Search Strategy**
- Start with alpha=0.5 and tune based on results
- Use different alpha values for different query types
- Consider query expansion for sparse search

### 4. **Maintenance**
```sql
-- Regular maintenance tasks
-- Vacuum and analyze tables
VACUUM ANALYZE documents;
VACUUM ANALYZE dense_embeddings;
VACUUM ANALYZE sparse_embeddings;

-- Refresh materialized views
REFRESH MATERIALIZED VIEW CONCURRENTLY document_search;

-- Monitor index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### 5. **Monitoring Queries**
```sql
-- Find slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
WHERE query LIKE '%embedding%'
ORDER BY mean_time DESC
LIMIT 10;

-- Check index efficiency
SELECT 
    indexrelname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    CASE 
        WHEN idx_scan = 0 THEN 0
        ELSE idx_tup_fetch::float / idx_scan 
    END as avg_tuples_per_scan
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;
```

## Troubleshooting

### Common Issues

1. **Slow dense search**
   - Increase `lists` parameter for IVFFlat index
   - Consider using HNSW index for better recall
   - Ensure vectors are normalized

2. **Slow sparse search**
   - Add more specific indexes on token_id
   - Consider partitioning for very large datasets
   - Use materialized views for common queries

3. **Memory issues**
   - Adjust `work_mem` and `shared_buffers`
   - Use connection pooling effectively
   - Consider partial indexes

4. **Inconsistent results**
   - Ensure proper normalization of scores
   - Check for NULL values in embeddings
   - Verify alpha parameter tuning

## Conclusion

This hybrid approach with PostgreSQL + pgvector provides:
- **Reliability** of PostgreSQL
- **Performance** through proper indexing
- **Flexibility** to tune dense vs sparse contribution
- **Scalability** to millions of documents
- **Integration** with existing PostgreSQL infrastructure

While it requires more manual implementation than purpose-built vector databases, it offers unmatched reliability and the ability to leverage PostgreSQL's mature ecosystem.