#!/usr/bin/env python3
"""
Example of using LLM Service with Langgraph.
"""
import os
import sys
from pathlib import Path
from typing import TypedDict, Annotated, Sequence
import operator

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END


# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    summary: str
    analysis_complete: bool


def create_llm(model_tier: str = "medium") -> ChatOpenAI:
    """Create an LLM instance connected to our service."""
    return ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
        model=model_tier,
        temperature=0.3
    )


def analyze_node(state: AgentState) -> dict:
    """Analyze the user's input."""
    llm = create_llm("medium")
    
    # Get the last user message
    last_message = state["messages"][-1].content
    
    # Analyze the message
    analysis_prompt = f"""Analyze this user message and determine its intent and key topics:
    
    User message: {last_message}
    
    Provide a brief analysis."""
    
    response = llm.invoke(analysis_prompt)
    
    return {
        "messages": [AIMessage(content=f"Analysis: {response.content}")],
        "analysis_complete": True
    }


def summarize_node(state: AgentState) -> dict:
    """Create a summary of the conversation."""
    llm = create_llm("light")  # Use light model for simple summary
    
    # Create conversation history
    history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
    
    summary_prompt = f"""Summarize this conversation in 2-3 sentences:
    
    {history}"""
    
    response = llm.invoke(summary_prompt)
    
    return {
        "summary": response.content,
        "messages": [AIMessage(content=f"Summary: {response.content}")]
    }


def respond_node(state: AgentState) -> dict:
    """Generate a final response."""
    llm = create_llm("heavy")  # Use heavy model for best response
    
    # Get context from previous nodes
    last_user_message = next(
        (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        ""
    )
    
    response_prompt = f"""Based on the analysis and summary, provide a helpful response to the user.
    
    User's original message: {last_user_message}
    Summary: {state.get('summary', 'No summary available')}
    
    Provide a comprehensive and helpful response."""
    
    response = llm.invoke(response_prompt)
    
    return {
        "messages": [AIMessage(content=response.content)]
    }


def should_continue(state: AgentState) -> str:
    """Determine if we should continue to summary or end."""
    if state.get("analysis_complete", False):
        return "summarize"
    return END


def create_agent_graph():
    """Create the langgraph workflow."""
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("respond", respond_node)
    
    # Define the flow
    workflow.set_entry_point("analyze")
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "summarize": "summarize",
            END: END
        }
    )
    workflow.add_edge("summarize", "respond")
    workflow.add_edge("respond", END)
    
    # Compile the graph
    return workflow.compile()


def test_simple_workflow():
    """Test a simple workflow."""
    print("Testing simple Langgraph workflow...")
    print("=" * 50)
    
    # Create the graph
    app = create_agent_graph()
    
    # Test input
    initial_state = {
        "messages": [HumanMessage(content="I need help planning a trip to Japan. I'm interested in technology and traditional culture.")],
        "summary": "",
        "analysis_complete": False
    }
    
    # Run the graph
    print("Running workflow...")
    result = app.invoke(initial_state)
    
    # Display results
    print("\nWorkflow Results:")
    print("-" * 30)
    for i, msg in enumerate(result["messages"]):
        print(f"\nStep {i + 1} ({msg.type}):")
        print(msg.content)
    
    if result.get("summary"):
        print(f"\nFinal Summary: {result['summary']}")


def test_multi_turn_conversation():
    """Test a multi-turn conversation with state management."""
    print("\n\nTesting multi-turn conversation...")
    print("=" * 50)
    
    # Simple chat graph
    class ChatState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        turn_count: int
    
    def chat_node(state: ChatState) -> dict:
        """Simple chat response node."""
        llm = create_llm("medium")
        
        # Get conversation history
        history = state["messages"]
        
        # Generate response
        response = llm.invoke(history)
        
        return {
            "messages": [response],
            "turn_count": state.get("turn_count", 0) + 1
        }
    
    # Create simple chat graph
    chat_workflow = StateGraph(ChatState)
    chat_workflow.add_node("chat", chat_node)
    chat_workflow.set_entry_point("chat")
    chat_workflow.add_edge("chat", END)
    
    chat_app = chat_workflow.compile()
    
    # Simulate conversation
    state = {
        "messages": [
            HumanMessage(content="Hi! I'm learning about machine learning.")
        ],
        "turn_count": 0
    }
    
    # First turn
    state = chat_app.invoke(state)
    print(f"Turn 1 - Assistant: {state['messages'][-1].content}")
    
    # Second turn
    state["messages"].append(HumanMessage(content="What's the difference between supervised and unsupervised learning?"))
    state = chat_app.invoke(state)
    print(f"\nTurn 2 - Assistant: {state['messages'][-1].content}")
    
    print(f"\nTotal turns: {state['turn_count']}")


def main():
    """Run all examples."""
    print("LLM Service Langgraph Example")
    print("=" * 50)
    print("Make sure the LLM service is running on http://localhost:8000")
    print("=" * 50)
    print()
    
    try:
        test_simple_workflow()
        test_multi_turn_conversation()
        
        print("\n\nAll Langgraph tests completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. The LLM service is running:")
        print("   cd /Users/patrickpark/Documents/Work/utils/llmservice")
        print("   python server.py")
        print("2. Langgraph is installed:")
        print("   pip install langgraph langchain-openai")


if __name__ == "__main__":
    main()