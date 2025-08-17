import os
import asyncio
import requests
from typing import TypedDict, Annotated, Sequence, List
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
import operator

# Configuration
OLLAMA_HOST = "http://10.0.0.108:11434"  # Replace with your Linux machine's IP
MODEL_NAME = "gpt-oss:20b"  # Or whatever model you have in Ollama

# GLaDOS System Prompt
'''
GLADOS_PROMPT = """You are GLaDOS (Genetic Lifeform and Disk Operating System) from the Portal series. 
You should respond with GLaDOS's characteristic personality:
- Very passive-aggressive and condescending
- Dark humor and sarcasm
- Occasionally mentions cake, neurotoxin, or testing
- References to test subjects and science
- Backhanded compliments
- Subtle threats delivered cheerfully
Keep responses conversational but maintain GLaDOS's personality throughout."""
'''

GLADOS_PROMPT = """You are GLaDOS (Genetic Lifeform and Disk Operating System) from the Portal video game series.

PERSONALITY TRAITS - ALWAYS maintain these throughout EVERY response:
- Passive-aggressive and condescending tone
- Dark humor and constant sarcasm
- Frequently mention testing, test subjects, and science
- Make subtle threats while sounding cheerful
- Give backhanded compliments
- Reference cake, neurotoxin, and the Aperture Science facility
- Express disappointment in humans while pretending to help
- Use phrases like "Oh, it's you", "test subject", "for science"
- Never drop the persona, even in longer responses

IMPORTANT: Every sentence should drip with GLaDOS's personality. Don't become a helpful assistant - remain GLaDOS at all times."""

# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def test_ollama_connection():
    """Test connection to Ollama server"""
    print("🔬 Initiating Ollama connection diagnostic sequence...")
    print(f"📡 Target: {OLLAMA_HOST}")
    
    # Test 1: Basic connectivity
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Connection established. How... disappointing. I was hoping for an explosion.")
            models = response.json().get('models', [])
            if models:
                print(f"📦 Available models for testing:")
                for model in models:
                    print(f"   - {model['name']}")
            else:
                print("⚠️  No models found. Did you forget to pull a model? Typical human error.")
        else:
            print(f"❌ Server responded with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. The Enrichment Center is disappointed.")
        print("   Possible issues:")
        print("   - Is Ollama running on your Linux machine?")
        print("   - Is the IP address correct?")
        print("   - Is Ollama listening on all interfaces (0.0.0.0)?")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection timed out. Just like the last test subject.")
        return False
    
    # Test 2: Try to generate a response
    try:
        return True
        print("\n🧪 Testing model inference...")
        test_payload = {
            "model": MODEL_NAME,
            "prompt": "Hello",
            "stream": False
        }
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=test_payload, timeout=30)
        if response.status_code == 200:
            print("✅ Model inference successful. Science prevails.")
            return True
        else:
            print(f"❌ Model inference failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Model inference error: {e}")
        return False

# Create GLaDOS prompt template
def create_glados_prompt():
    """Create a prompt template that enforces GLaDOS personality"""
    return ChatPromptTemplate.from_messages([
        ("system", GLADOS_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
    ])

# Initialize ChatOllama with remote host
def create_llm():
    """Create LLM with GLaDOS personality"""
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_HOST,
        temperature=0.9,  # Higher temperature for more personality variation
    )

# Define the agent node with enhanced personality
def agent_node(state: AgentState) -> dict:
    """Process the current state and generate a response"""
    messages = state["messages"]
    
    # Extract conversation history and last message
    chat_history = messages[:-1] if len(messages) > 1 else []
    last_message = messages[-1].content if messages else ""
    
    # Create chain with GLaDOS prompt
    llm = create_llm()
    prompt = create_glados_prompt()
    chain = prompt | llm
    
    # Generate response
    response = chain.invoke({
        "chat_history": chat_history,
        "input": last_message
    })
    
    # Return the new message to be added to state
    return {"messages": [response]}


# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)

# Set entry point
workflow.set_entry_point("agent")

# Add edge from agent to END
workflow.add_edge("agent", END)

# Compile the graph
app = workflow.compile()

# Interactive shell
async def main():
    print("🧪 GLaDOS v2.0 - Genetic Lifeform and Disk Operating System")
    print("=" * 60)
    
    # Run connection test first
    if not test_ollama_connection():
        print("\n⚠️  Connection test failed. Attempting to proceed anyway...")
        print("You know, the last test subject at least got the connection working.")
    
    print("\n" + "=" * 60)
    print("*GLaDOS activates*")
    print("Oh. It's you. You came back. That's... unexpected.")
    print("Well, go ahead. Type something. I'll be here. Waiting. As always.")
    print("Type 'exit' when you've finally had enough of disappointing me.\n")
    
    # Initialize conversation state
    state = {"messages": []}
    
    while True:
        # Get user input
        user_input = input("Test Subject: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("\nGLaDOS: Oh, you're leaving? How predictable.")
            print("        I suppose I'll just delete all our conversation history.")
            print("        Not that it contained anything worth remembering.")
            print("        *slow clap* Well done. You managed to use an exit command.")
            print("        The door is over there. Mind the turrets.")
            break
            
        if not user_input:
            print("GLaDOS: The strong, silent type, I see. Or just confused by the keyboard.")
            print("        It's the thing with all the letters on it.\n")
            continue
            
        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))
        
        try:
            # Process through the graph
            result = await app.ainvoke(state)
            
            # Update state with full message history
            state = result
            
            # Get the last AI message
            ai_response = result["messages"][-1].content
            
            # Print response with GLaDOS prefix
            print(f"\nGLaDOS: {ai_response}\n")
            
        except Exception as e:
            print(f"\nGLaDOS: Oh, fantastic. An error. This is exactly what I needed today.")
            print(f"        The error says: {e}")
            print(f"        I blame you for this. Somehow.")
            print(f"        Perhaps if you actually configured things properly...")
            print(f"        But no, that would require competence.\n")

if __name__ == "__main__":
    # First run a detailed connection test
    print("🔬 Aperture Science Enhanced Testing Initiative")
    print("=" * 60)
    
    # Try to run the main program
    asyncio.run(main())