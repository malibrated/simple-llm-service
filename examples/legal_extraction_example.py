#!/usr/bin/env python3
"""
Example of using LLM Service for legal entity and relationship extraction.
This demonstrates how to migrate from the old LLMFactory to the new service.
"""
import asyncio
import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


@dataclass
class ExtractionResult:
    """Result from entity/relationship extraction."""
    document_id: int
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    processing_time: float
    model_used: str


class LegalExtractor:
    """Legal entity and relationship extractor using LLM Service."""
    
    # Entity extraction prompt
    ENTITY_PROMPT = """You are a legal entity extractor. Extract all legal entities from the text.

Entity types:
- case: Court cases (e.g., "Smith v. Jones", "Miranda v. Arizona")
- statute: Laws and statutes (e.g., "28 U.S.C. § 1331", "CCP § 410.10")
- court: Courts and jurisdictions (e.g., "U.S. Supreme Court", "9th Circuit")
- concept: Legal concepts and doctrines (e.g., "personal jurisdiction", "due process")
- rule: Court rules and procedures (e.g., "FRCP 12(b)(6)", "Rule 56")
- test: Legal tests and standards (e.g., "minimum contacts test")

Return JSON with this structure:
{
  "entities": [
    {"name": "entity name", "type": "entity type", "context": "brief context"}
  ]
}

Text to analyze:
"""

    # Relationship extraction prompt
    RELATIONSHIP_PROMPT = """You are a legal relationship extractor. Identify relationships between the given entities.

Relationship types:
- cites: One case/document cites another
- establishes: Case establishes a principle or test
- interprets: Case interprets a statute
- applies: Case applies a test or doctrine
- overrules: Case overrules another
- defines: Case/statute defines a concept
- requires: Statute/rule requires something

Entities in this document:
{entities}

Text:
{text}

Return JSON with this structure:
{{
  "relationships": [
    {{"source": "source entity", "target": "target entity", "type": "relationship type", "context": "explanation"}}
  ]
}}"""
    
    def __init__(self, service_url: str = "http://localhost:8000/v1"):
        """Initialize with LLM service URL."""
        # Light model for classification
        self.classifier = ChatOpenAI(
            base_url=service_url,
            api_key="not-needed",
            model="light",
            temperature=0.1,
            max_tokens=50
        )
        
        # Medium model for entity extraction
        self.entity_extractor = ChatOpenAI(
            base_url=service_url,
            api_key="not-needed",
            model="medium",
            temperature=0.1,
            max_tokens=2048,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        # Heavy model for relationship extraction
        self.relationship_extractor = ChatOpenAI(
            base_url=service_url,
            api_key="not-needed",
            model="heavy",
            temperature=0.3,
            max_tokens=3000,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    
    async def classify_document(self, text: str) -> str:
        """Classify document type using light model."""
        prompt = f"Classify this legal document as one of: case, statute, rule, treatise, brief. Document: {text[:200]}..."
        response = await self.classifier.ainvoke(prompt)
        return response.content.strip().lower()
    
    async def extract_entities(self, text: str, max_length: int = 4000) -> List[Dict[str, Any]]:
        """Extract legal entities from text."""
        # Truncate text if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        prompt = self.ENTITY_PROMPT + text
        
        try:
            response = await self.entity_extractor.ainvoke(prompt)
            result = json.loads(response.content)
            return result.get("entities", [])
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response was: {response.content}")
            return []
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []
    
    async def extract_relationships(self, text: str, entities: List[Dict[str, Any]], 
                                  max_length: int = 6000) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        if len(entities) < 2:
            return []
        
        # Format entities for prompt
        entity_names = [e["name"] for e in entities]
        entities_str = "\n".join(f"- {name}" for name in entity_names[:20])  # Limit to 20 entities
        
        # Truncate text if needed
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        prompt = self.RELATIONSHIP_PROMPT.format(entities=entities_str, text=text)
        
        try:
            response = await self.relationship_extractor.ainvoke(prompt)
            result = json.loads(response.content)
            
            # Validate relationships (ensure entities exist)
            valid_relationships = []
            for rel in result.get("relationships", []):
                if rel["source"] in entity_names and rel["target"] in entity_names:
                    valid_relationships.append(rel)
            
            return valid_relationships
        except Exception as e:
            print(f"Error extracting relationships: {e}")
            return []
    
    async def process_document(self, doc_id: int, text: str) -> ExtractionResult:
        """Process a complete document for entities and relationships."""
        start_time = datetime.now()
        
        # Step 1: Classify document
        doc_type = await self.classify_document(text)
        print(f"Document {doc_id} classified as: {doc_type}")
        
        # Step 2: Extract entities
        entities = await self.extract_entities(text)
        print(f"Found {len(entities)} entities")
        
        # Step 3: Extract relationships (if entities found)
        relationships = []
        if len(entities) >= 2:
            relationships = await self.extract_relationships(text, entities)
            print(f"Found {len(relationships)} relationships")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ExtractionResult(
            document_id=doc_id,
            entities=entities,
            relationships=relationships,
            processing_time=processing_time,
            model_used=f"{doc_type}-extraction"
        )


async def demo_extraction():
    """Demonstrate extraction on sample legal text."""
    # Sample legal text
    sample_text = """
    In International Shoe Co. v. Washington, 326 U.S. 310 (1945), the United States Supreme Court 
    established the "minimum contacts" test for determining whether a state court has personal 
    jurisdiction over a non-resident defendant. The Court held that due process requires only that 
    the defendant have certain minimum contacts with the forum state such that the maintenance of 
    the suit does not offend traditional notions of fair play and substantial justice.
    
    This landmark decision overruled Pennoyer v. Neff, 95 U.S. 714 (1878), which had required 
    physical presence within the state for jurisdiction. The minimum contacts test has been further 
    refined in cases such as World-Wide Volkswagen Corp. v. Woodson, 444 U.S. 286 (1980), which 
    clarified that the defendant's conduct and connection with the forum state must be such that 
    he should reasonably anticipate being haled into court there.
    
    Under FRCP Rule 4(k), federal courts generally have personal jurisdiction over a defendant who 
    is subject to the jurisdiction of a state court of general jurisdiction in the state where the 
    district court is located. This rule incorporates the International Shoe standard into federal 
    practice.
    """
    
    # Create extractor
    extractor = LegalExtractor()
    
    # Process the document
    print("Processing sample legal text...")
    print("=" * 60)
    
    result = await extractor.process_document(1, sample_text)
    
    # Display results
    print(f"\nProcessing completed in {result.processing_time:.2f} seconds")
    print(f"\nEntities found ({len(result.entities)}):")
    for entity in result.entities:
        print(f"  - {entity['name']} ({entity['type']})")
        print(f"    Context: {entity['context'][:100]}...")
    
    print(f"\nRelationships found ({len(result.relationships)}):")
    for rel in result.relationships:
        print(f"  - {rel['source']} {rel['type']} {rel['target']}")
        print(f"    Context: {rel['context'][:100]}...")


async def batch_extraction_demo():
    """Demonstrate batch processing of multiple documents."""
    documents = [
        {
            "id": 1,
            "text": "In Marbury v. Madison, 5 U.S. 137 (1803), Chief Justice John Marshall established the principle of judicial review."
        },
        {
            "id": 2,
            "text": "California Code of Civil Procedure § 410.10 provides that a court of this state may exercise jurisdiction on any basis not inconsistent with the Constitution."
        },
        {
            "id": 3,
            "text": "The purposeful availment requirement ensures that a defendant will not be haled into a jurisdiction solely as a result of random, fortuitous, or attenuated contacts."
        }
    ]
    
    extractor = LegalExtractor()
    
    print("\nBatch Processing Demo")
    print("=" * 60)
    
    # Process all documents concurrently
    tasks = [extractor.process_document(doc["id"], doc["text"]) for doc in documents]
    results = await asyncio.gather(*tasks)
    
    # Summary
    total_entities = sum(len(r.entities) for r in results)
    total_relationships = sum(len(r.relationships) for r in results)
    total_time = sum(r.processing_time for r in results)
    
    print(f"\nBatch Processing Summary:")
    print(f"  Documents processed: {len(results)}")
    print(f"  Total entities: {total_entities}")
    print(f"  Total relationships: {total_relationships}")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Average time per document: {total_time/len(results):.2f}s")


def main():
    """Run extraction demos."""
    print("Legal Entity and Relationship Extraction Demo")
    print("Using LLM Service at http://localhost:8000")
    print("=" * 60)
    
    # Check if service is available
    import requests
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("Error: LLM Service is not healthy")
            return
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to LLM Service")
        print("Please start the service first:")
        print("  cd /Users/patrickpark/Documents/Work/utils/llmservice")
        print("  ./start_service.sh")
        return
    
    # Run demos
    asyncio.run(demo_extraction())
    asyncio.run(batch_extraction_demo())


if __name__ == "__main__":
    main()