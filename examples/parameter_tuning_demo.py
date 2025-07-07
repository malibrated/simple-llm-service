#!/usr/bin/env python3
"""
Demo showing how different requesters can tune parameters independently.
The same loaded model serves all requests with different behaviors.
"""
import asyncio
import json
import time
from typing import List, Dict, Any
import aiohttp
from datetime import datetime


class MultiUserSimulation:
    """Simulate multiple users with different parameter preferences."""
    
    def __init__(self, service_url: str = "http://localhost:8000/v1"):
        self.service_url = service_url
    
    async def researcher_request(self, prompt: str) -> Dict[str, Any]:
        """Researcher wants accurate, deterministic results."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "medium",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,      # Very low randomness
                "max_tokens": 2000,      # Detailed responses
                "top_k": 10,            # Focused selection
                "repeat_penalty": 1.05,  # Avoid repetition
                "seed": 42              # Reproducible results
            }
            
            async with session.post(f"{self.service_url}/chat/completions", json=payload) as resp:
                return await resp.json()
    
    async def creative_writer_request(self, prompt: str) -> Dict[str, Any]:
        """Creative writer wants varied, imaginative outputs."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "medium",  # Same model!
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.9,      # High randomness
                "max_tokens": 1500,      # Medium length
                "top_p": 0.95,          # Diverse vocabulary
                "repeat_penalty": 1.2,   # Strong repetition penalty
                "seed": int(time.time()) # Different each time
            }
            
            async with session.post(f"{self.service_url}/chat/completions", json=payload) as resp:
                return await resp.json()
    
    async def classifier_request(self, prompt: str) -> Dict[str, Any]:
        """Classifier wants fast, deterministic labels."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "light",  # Fast model
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,      # Completely deterministic
                "max_tokens": 10,        # Just the label
                "top_k": 1,             # Greedy decoding
            }
            
            async with session.post(f"{self.service_url}/chat/completions", json=payload) as resp:
                return await resp.json()
    
    async def analyst_request(self, prompt: str) -> Dict[str, Any]:
        """Analyst wants balanced, comprehensive analysis."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "heavy",  # Best model for complex analysis
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,      # Balanced
                "max_tokens": 4000,      # Long, detailed response
                "top_p": 0.9,           # Slightly focused
                "repeat_penalty": 1.1,   # Moderate repetition control
            }
            
            async with session.post(f"{self.service_url}/chat/completions", json=payload) as resp:
                return await resp.json()


async def demo_concurrent_users():
    """Show multiple users making requests with different parameters simultaneously."""
    print("Multi-User Parameter Tuning Demo")
    print("=" * 60)
    
    sim = MultiUserSimulation()
    
    # Different prompts for different users
    prompts = {
        "researcher": "What are the key principles established in International Shoe v. Washington?",
        "writer": "Write a creative opening for a legal thriller involving corporate espionage.",
        "classifier": "Classify this document type: Motion for Summary Judgment",
        "analyst": "Analyze the implications of AI-generated content on copyright law."
    }
    
    print("\nSending concurrent requests with different parameters...")
    start_time = time.time()
    
    # Make all requests concurrently
    results = await asyncio.gather(
        sim.researcher_request(prompts["researcher"]),
        sim.creative_writer_request(prompts["writer"]),
        sim.classifier_request(prompts["classifier"]),
        sim.analyst_request(prompts["analyst"]),
        return_exceptions=True
    )
    
    elapsed = time.time() - start_time
    
    # Display results
    users = ["Researcher", "Creative Writer", "Classifier", "Analyst"]
    for user, result in zip(users, results):
        print(f"\n{user} Result:")
        print("-" * 40)
        
        if isinstance(result, Exception):
            print(f"Error: {result}")
        else:
            response_text = result["choices"][0]["message"]["content"]
            tokens = result["usage"]["total_tokens"]
            print(f"Response ({tokens} tokens): {response_text[:200]}...")


async def demo_parameter_effects():
    """Show how the same prompt gives different results with different parameters."""
    print("\n\nParameter Effects Demo")
    print("=" * 60)
    
    prompt = "Explain the concept of legal precedent"
    
    async with aiohttp.ClientSession() as session:
        configs = [
            {
                "name": "Conservative (T=0.1)",
                "temperature": 0.1,
                "max_tokens": 100,
                "top_k": 10
            },
            {
                "name": "Balanced (T=0.5)",
                "temperature": 0.5,
                "max_tokens": 100,
                "top_p": 0.9
            },
            {
                "name": "Creative (T=0.9)",
                "temperature": 0.9,
                "max_tokens": 100,
                "top_p": 0.95,
                "repeat_penalty": 1.3
            }
        ]
        
        for config in configs:
            payload = {
                "model": "medium",
                "messages": [{"role": "user", "content": prompt}],
                **{k: v for k, v in config.items() if k != "name"}
            }
            
            async with session.post(
                "http://localhost:8000/v1/chat/completions",
                json=payload
            ) as resp:
                result = await resp.json()
                
            print(f"\n{config['name']}:")
            print("-" * 40)
            print(result["choices"][0]["message"]["content"])


async def demo_dynamic_adjustment():
    """Show dynamic parameter adjustment based on task complexity."""
    print("\n\nDynamic Parameter Adjustment Demo")
    print("=" * 60)
    
    class AdaptiveLLM:
        def __init__(self):
            self.service_url = "http://localhost:8000/v1/chat/completions"
        
        async def analyze_complexity(self, text: str) -> float:
            """Simple complexity scoring based on text characteristics."""
            word_count = len(text.split())
            unique_words = len(set(text.lower().split()))
            
            # Simple heuristics
            if word_count < 10:
                return 0.2
            elif "analyze" in text.lower() or "explain" in text.lower():
                return 0.8
            elif unique_words / word_count > 0.8:
                return 0.6
            else:
                return 0.4
        
        async def generate(self, prompt: str) -> str:
            """Generate with dynamically adjusted parameters."""
            complexity = await self.analyze_complexity(prompt)
            
            # Adjust parameters based on complexity
            if complexity < 0.3:
                params = {
                    "model": "light",
                    "temperature": 0.1,
                    "max_tokens": 50
                }
            elif complexity < 0.6:
                params = {
                    "model": "medium",
                    "temperature": 0.3,
                    "max_tokens": 500
                }
            else:
                params = {
                    "model": "heavy",
                    "temperature": 0.5,
                    "max_tokens": 2000
                }
            
            params["messages"] = [{"role": "user", "content": prompt}]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.service_url, json=params) as resp:
                    result = await resp.json()
                    
            return result["choices"][0]["message"]["content"], complexity, params["model"]
    
    adaptive = AdaptiveLLM()
    
    test_prompts = [
        "What is tort law?",  # Simple
        "Compare negligence and strict liability in product liability cases.",  # Medium
        "Analyze the evolution of privacy rights in the digital age, considering Fourth Amendment jurisprudence and modern surveillance technologies."  # Complex
    ]
    
    for prompt in test_prompts:
        response, complexity, model_used = await adaptive.generate(prompt)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"Complexity Score: {complexity:.2f}")
        print(f"Model Used: {model_used}")
        print(f"Response: {response[:150]}...")


async def demo_structured_output_params():
    """Show how grammar parameters ensure structured output."""
    print("\n\nStructured Output with Grammar Demo")
    print("=" * 60)
    
    # Grammar for extracting case citations
    citation_grammar = '''
root ::= object
object ::= "{" ws "\\"citations\\"" ws ":" ws array ws "}"
array ::= "[" ws "]" | "[" ws citation ("," ws citation)* ws "]"
citation ::= "{" ws "\\"case\\"" ws ":" ws string ws "," ws "\\"year\\"" ws ":" ws number ws "}"
string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""
number ::= [0-9]+
ws ::= [ \\t\\n]*
'''
    
    prompt = "Extract case citations from: The court relied on Brown v. Board of Education (1954) and Miranda v. Arizona (1966) in reaching its decision."
    
    async with aiohttp.ClientSession() as session:
        # With grammar - structured output
        payload_with_grammar = {
            "model": "medium",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 200,
            "grammar": citation_grammar
        }
        
        # Without grammar - free text
        payload_without_grammar = {
            "model": "medium",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 200
        }
        
        print("With Grammar Constraint:")
        async with session.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload_with_grammar
        ) as resp:
            result = await resp.json()
            response = result["choices"][0]["message"]["content"]
            print(response)
            
            # Verify it's valid JSON
            try:
                parsed = json.loads(response)
                print("✓ Valid JSON structure")
            except:
                print("✗ Invalid JSON")
        
        print("\nWithout Grammar Constraint:")
        async with session.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload_without_grammar
        ) as resp:
            result = await resp.json()
            response = result["choices"][0]["message"]["content"]
            print(response[:200])


async def main():
    """Run all demos."""
    try:
        # Check service health
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health") as resp:
                if resp.status != 200:
                    print("Error: LLM Service is not healthy")
                    return
    except:
        print("Error: Cannot connect to LLM Service at http://localhost:8000")
        print("Please start the service first with: ./start_service.sh")
        return
    
    # Run demos
    await demo_concurrent_users()
    await demo_parameter_effects()
    await demo_dynamic_adjustment()
    await demo_structured_output_params()
    
    print("\n\nAll demos completed!")
    print("\nKey Takeaways:")
    print("- Same model serves all requests with different parameters")
    print("- Parameters are tuned per-request, not per-model")
    print("- Enables multi-tenant usage with personalized behavior")
    print("- Grammar constraints ensure structured output when needed")


if __name__ == "__main__":
    asyncio.run(main())