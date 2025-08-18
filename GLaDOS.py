import os
import asyncio
import requests
from typing import TypedDict, Annotated, Sequence, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
import operator
import pygame
import io
import threading
import logging

# Configuration
OLLAMA_HOST = "http://10.0.0.108:11434"  # Replace with your Linux machine's IP
TTS_HOST = "http://10.0.0.108:8001"     # Chatterbox TTS server
MODEL_NAME = "gpt-oss:20b"  # Or whatever model you have in Ollama
ENABLE_TTS = True  # Toggle TTS on/off

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

# Initialize pygame for audio playback
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class GLaDOSTTS:
    """Handle text-to-speech for GLaDOS"""
    
    def __init__(self, tts_host: str):
        self.tts_host = tts_host
        self.is_playing = False
        
    def speak(self, text: str) -> None:
        """Generate and play GLaDOS speech"""
        if not ENABLE_TTS:
            return
            
        def _play_audio():
            try:
                self.is_playing = True
                
                # Request TTS
                response = requests.post(
                    f"{self.tts_host}/tts",
                    json={
                        "text": text,
                        "exaggeration": 0.7,  # GLaDOS dramatic style
                        "cfg_weight": 0.3,    # Deliberate pacing
                        "use_glados_voice": True
                    },
                    timeout=30  # TTS can take time
                )
                
                if response.status_code == 200:
                    # Load audio from response
                    audio_data = io.BytesIO(response.content)
                    sound = pygame.mixer.Sound(audio_data)
                    
                    # Play and wait for completion
                    sound.play()
                    while pygame.mixer.get_busy():
                        pygame.time.wait(100)
                else:
                    logger.error(f"TTS request failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"TTS playback error: {e}")
            finally:
                self.is_playing = False
        
        # Play audio in background thread to not block
        audio_thread = threading.Thread(target=_play_audio)
        audio_thread.daemon = True
        audio_thread.start()
    
    def wait_for_speech(self):
        """Wait for current speech to finish"""
        while self.is_playing:
            pygame.time.wait(100)

def test_connections():
    global ENABLE_TTS
    """Test connections to both Ollama and Chatterbox"""
    print("🔬 Initiating Aperture Science systems diagnostic...")
    
    # Test Ollama
    print(f"\n📡 Testing Ollama at: {OLLAMA_HOST}")
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama connection established")
            models = response.json().get('models', [])
            if models:
                print(f"📦 Available models: {', '.join(m['name'] for m in models)}")
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False
    
    # Test Chatterbox TTS
    print(f"\n🔊 Testing Chatterbox TTS at: {TTS_HOST}")
    try:
        response = requests.get(f"{TTS_HOST}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ TTS connection established")
            print(f"   Device: {health.get('device', 'unknown')}")
            print(f"   GLaDOS voice: {'available' if health.get('glados_voice_available') else 'not found'}")
            
            # Test TTS generation
            if ENABLE_TTS:
                print("🧪 Testing audio generation...")
                test_response = requests.post(
                    f"{TTS_HOST}/tts",
                    json={"text": "Testing. Testing. One. Two. Three."},
                    timeout=10
                )
                if test_response.status_code == 200:
                    print("✅ TTS generation successful")
                else:
                    print(f"⚠️  TTS generation failed: {test_response.status_code}")
        else:
            print(f"❌ TTS responded with status {response.status_code}")
            ENABLE_TTS = False
    except Exception as e:
        print(f"⚠️  TTS connection failed: {e}")
        print("   Continuing without voice output...")
        ENABLE_TTS = False
    
    return True

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
    
    # Run connection tests
    if not test_connections():
        print("\n⚠️  System check failed. Proceeding anyway...")
        print("I suppose we'll have to make do with suboptimal conditions.")
        print("How very... human of you.")
    
    # Initialize TTS
    tts = GLaDOSTTS(TTS_HOST) if ENABLE_TTS else None

    print("\n" + "=" * 60)
    print("*GLaDOS activates*")
    # Opening speech
    opening = "Oh. It's you. You came back. That's... unexpected. Well, go ahead. Type something. I'll be here. Waiting. As always."
    print(f"GLaDOS: {opening}")
    if tts:
        tts.speak(opening)
    print("\nType 'exit' when you've finally had enough of disappointing me.\n")

    # Initialize conversation state
    state = {"messages": []}
    
    while True:
        # Wait for any ongoing speech to finish before accepting input
        if tts:
            tts.wait_for_speech()
        
        # Get user input
        user_input = input("Test Subject: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            farewell = "Oh, you're leaving? How predictable. I suppose I'll just delete all our conversation history. Not that it contained anything worth remembering. *slow clap* Well done. You managed to use an exit command. The door is over there. Mind the turrets."
            print(f"\nGLaDOS: {farewell}")
            if tts:
                tts.speak(farewell)
                tts.wait_for_speech()
            break
            
        if not user_input:
            empty_response = "The strong, silent type, I see. Or just confused by the keyboard. It's the thing with all the letters on it."
            print(f"GLaDOS: {empty_response}\n")
            if tts:
                tts.speak(empty_response)
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
            
            # Print response
            print(f"\nGLaDOS: {ai_response}\n")
            
            # Speak response
            if tts:
                tts.speak(ai_response)
            
        except Exception as e:
            error_response = f"Oh, fantastic. An error. This is exactly what I needed today. The error says: {e}. I blame you for this. Somehow. Perhaps if you actually configured things properly... But no, that would require competence."
            print(f"\nGLaDOS: {error_response}\n")
            if tts:
                tts.speak(error_response)

if __name__ == "__main__":
    # First run a detailed connection test
    print("🔬 Aperture Science Enhanced Testing Initiative")
    print("=" * 60)
    
    # Try to run the main program
    asyncio.run(main())