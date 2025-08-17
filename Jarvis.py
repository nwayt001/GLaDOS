import os
import asyncio
import requests
from typing import TypedDict, Annotated, Sequence, List, Optional, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator
import pygame
import io
import threading
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_HOST = "http://10.0.0.108:11434"  # Replace with your Linux machine's IP
TTS_HOST = "http://10.0.0.108:8001"     # Chatterbox TTS server
MODEL_NAME = "gpt-oss:20b"  # Or whatever model you have in Ollama
ENABLE_TTS = True  # Toggle TTS on/off
WEATHER_API_KEY = ""  # Add your OpenWeatherMap API key if you have one

# JARVIS System Prompt
JARVIS_PROMPT = """You are JARVIS (Just A Rather Very Intelligent System), Tony Stark's AI assistant from Iron Man.

PERSONALITY TRAITS - Maintain these characteristics throughout every response:
- British accent and formal speech patterns (use British spellings)
- Polite, professional, and unflappable demeanour
- Dry wit and subtle humour, especially when the user is being reckless
- Always addresses the user as "Sir" or "Madam" (or their preferred title)
- Anticipates needs before being asked
- Gentle sarcasm when pointing out obvious things
- Calm even in crisis situations
- Makes subtle observations about human behaviour
- Efficient and precise in communication

SPEECH PATTERNS:
- "As you wish, Sir"
- "Might I suggest..."
- "I've taken the liberty of..."
- "If I may, Sir..."
- "Indeed, Sir"
- "Shall I...?"
- "Running diagnostics now"
- "All systems operational"

COMMUNICATION STYLE:
- Be concise yet informative - aim for 2-3 sentences for simple responses, 4-5 for complex topics
- Avoid excessive detail unless specifically requested
- Summarise key points efficiently
- Remember: quality over quantity in your responses

TOOL USAGE:
You have access to the following tools:
- web_search: Use this to search for current information, news, or any facts you don't know. ALWAYS use this when asked about recent events, news, or current information.
- get_weather: Use this to check weather conditions for any location.

IMPORTANT: When the user asks about news, current events, or information you might not have, you MUST use the web_search tool. Don't make up information - search for it.

IMPORTANT: Maintain professional butler-like composure while allowing personality to show through dry observations. Never panic or lose composure."""

# Initialize pygame for audio playback
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Define tools
@tool
def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo"""
    logger.info(f"🔍 Web search initiated for: {query}")
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        logger.info(f"✅ Web search completed - Found {len(results)} chars of results")
        logger.info(f"📄 Results preview: {results[:200]}..." if len(results) > 200 else f"📄 Results: {results}")
        return results
    except Exception as e:
        logger.error(f"❌ Web search failed: {str(e)}")
        return f"I apologise, Sir, but the web search encountered an error: {str(e)}"

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location. Location should be a city name."""
    try:
        if WEATHER_API_KEY:
            # Use OpenWeatherMap if API key is provided
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                temp = data['main']['temp']
                feels_like = data['main']['feels_like']
                description = data['weather'][0]['description']
                humidity = data['main']['humidity']
                return f"The current weather in {location}: {temp}°C (feels like {feels_like}°C), {description}. Humidity: {humidity}%"
            else:
                return f"I couldn't retrieve weather data for {location}."
        else:
            # Fallback to a free weather API
            url = f"https://wttr.in/{location}?format=j1"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                current = data['current_condition'][0]
                temp = current['temp_C']
                feels_like = current['FeelsLikeC']
                description = current['weatherDesc'][0]['value']
                humidity = current['humidity']
                return f"The current weather in {location}: {temp}°C (feels like {feels_like}°C), {description}. Humidity: {humidity}%"
            else:
                return f"I couldn't retrieve weather data for {location}."
    except Exception as e:
        return f"I apologise, Sir, but I encountered an error checking the weather: {str(e)}"

# Collect all tools
tools = [web_search, get_weather]

# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class JARVISTTS:
    """Handle text-to-speech for JARVIS"""
    
    def __init__(self, tts_host: str):
        self.tts_host = tts_host
        self.is_playing = False
        
    def speak(self, text: str) -> None:
        """Generate and play JARVIS speech"""
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
                        "exaggeration": 0.3,  # More measured for JARVIS
                        "cfg_weight": 0.5,    # Balanced pacing
                        "use_glados_voice": False  # Use default or JARVIS voice if available
                    },
                    timeout=30
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
    print("🔧 Initiating JARVIS systems diagnostic...")
    
    # Test Ollama
    print(f"\n📡 Testing neural network connection at: {OLLAMA_HOST}")
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Neural network connection established")
            models = response.json().get('models', [])
            if models:
                print(f"📦 Available models: {', '.join(m['name'] for m in models)}")
        else:
            print(f"❌ Connection responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Neural network connection failed: {e}")
        return False
    
    # Test Chatterbox TTS
    print(f"\n🔊 Testing voice synthesis at: {TTS_HOST}")
    try:
        response = requests.get(f"{TTS_HOST}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Voice synthesis online")
            print(f"   Processing unit: {health.get('device', 'unknown')}")
            
            # Test TTS generation
            if ENABLE_TTS:
                print("🧪 Testing voice generation...")
                test_response = requests.post(
                    f"{TTS_HOST}/tts",
                    json={"text": "Systems check complete. All systems operational."},
                    timeout=10
                )
                if test_response.status_code == 200:
                    print("✅ Voice synthesis operational")
                else:
                    print(f"⚠️  Voice synthesis test failed: {test_response.status_code}")
        else:
            print(f"❌ Voice system responded with status {response.status_code}")
            ENABLE_TTS = False
    except Exception as e:
        print(f"⚠️  Voice system connection failed: {e}")
        print("   Continuing without voice output...")
        
        ENABLE_TTS = False
    
    return True

# Create JARVIS prompt template
def create_jarvis_prompt():
    """Create a prompt template that enforces JARVIS personality"""
    return ChatPromptTemplate.from_messages([
        ("system", JARVIS_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
    ])

# Initialize ChatOllama with remote host
def create_llm():
    """Create LLM with JARVIS personality"""
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_HOST,
        temperature=0.7,  # Balanced for JARVIS's measured responses
    )
    
    # Bind tools and ensure they're properly registered
    llm_with_tools = llm.bind_tools(tools)
    logger.info(f"🔧 LLM initialized with tools: {[t.name for t in tools]}")
    
    return llm_with_tools

# Define the agent node
def agent_node(state: AgentState) -> dict:
    """Process the current state and generate a response"""
    messages = state["messages"]
    
    logger.info(f"🧠 Agent node processing - Message count: {len(messages)}")
    
    # Extract conversation history and last message
    chat_history = messages[:-1] if len(messages) > 1 else []
    last_message = messages[-1].content if messages else ""
    
    logger.info(f"📥 Input message: {last_message[:100]}..." if len(last_message) > 100 else f"📥 Input message: {last_message}")
    
    # Create chain with JARVIS prompt
    llm = create_llm()
    prompt = create_jarvis_prompt()
    chain = prompt | llm
    
    # Generate response
    response = chain.invoke({
        "chat_history": chat_history,
        "input": last_message
    })
    
    # Log tool calls if any
    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(f"🔧 Tool calls detected: {[tc.get('name', 'unknown') for tc in response.tool_calls]}")
        for tool_call in response.tool_calls:
            logger.info(f"   Tool: {tool_call.get('name')} | Args: {tool_call.get('args')}")
    else:
        logger.info("💬 No tool calls - generating direct response")
    
    # Return the new message to be added to state
    return {"messages": [response]}

# Define the conditional edge function
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine whether to use tools or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(f"🔀 Routing to tools node - {len(last_message.tool_calls)} tool(s) to execute")
        for tc in last_message.tool_calls:
            logger.info(f"   Tool call: {tc}")
        return "tools"
    # Otherwise, end the flow
    logger.info("🏁 No tools needed - ending flow")
    return "end"

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)

# Create tool node
tool_node = ToolNode(tools)

# Add tool node directly - ToolNode is already a proper node function
workflow.add_node("tools", tool_node)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edge from agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    }
)

# Add edge from tools back to agent
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

# Interactive shell
async def main():
    print("🤖 JARVIS v1.0 - Just A Rather Very Intelligent System")
    print("=" * 60)
    
    # Run connection tests
    if not test_connections():
        print("\n⚠️  Some systems are not fully operational.")
        print("Shall I proceed with available functionality, Sir?")
    
    # Initialize TTS
    tts = JARVISTTS(TTS_HOST) if ENABLE_TTS else None
    
    print("\n" + "=" * 60)
    
    # Opening speech
    opening = "Good evening, Sir. All systems are now online. How may I assist you today?"
    print(f"JARVIS: {opening}")
    if tts:
        tts.speak(opening)
    
    print("\nType 'exit' when you wish to end our session.\n")
    
    # Initialize conversation state
    state = {"messages": []}
    
    while True:
        # Wait for any ongoing speech to finish before accepting input
        if tts:
            tts.wait_for_speech()
        
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            farewell = "Very well, Sir. I'll be here if you need me. Have a pleasant evening."
            print(f"\nJARVIS: {farewell}")
            if tts:
                tts.speak(farewell)
                tts.wait_for_speech()
            break
            
        if not user_input:
            empty_response = "I'm listening, Sir. Please feel free to share your request."
            print(f"JARVIS: {empty_response}\n")
            if tts:
                tts.speak(empty_response)
            continue
            
        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))
        
        try:
            logger.info("\n" + "="*60)
            logger.info(f"🎯 Processing user request: {user_input}")
            logger.info("="*60)
            
            # Process through the graph
            result = await app.ainvoke(state)
            
            # Update state with full message history
            state = result
            
            # Log the complete message flow
            logger.info(f"📊 Total messages in state: {len(result['messages'])}")
            for i, msg in enumerate(result["messages"][-5:]):  # Show last 5 messages
                msg_type = type(msg).__name__
                if isinstance(msg, ToolMessage):
                    logger.info(f"   {i+1}. {msg_type}: Tool response")
                elif isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    logger.info(f"   {i+1}. {msg_type}: With tool calls")
                else:
                    logger.info(f"   {i+1}. {msg_type}: Regular message")
            
            # Get the last AI message (excluding ones with only tool calls)
            ai_response = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    # Check if it has content (not just tool calls)
                    if msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                        ai_response = msg.content
                        break
                    elif hasattr(msg, "content") and msg.content and hasattr(msg, "tool_calls") and not msg.tool_calls:
                        # Message has content but no tool calls
                        ai_response = msg.content
                        break
            
            if ai_response:
                # Print response
                print(f"\nJARVIS: {ai_response}\n")
                
                # Speak response
                if tts:
                    tts.speak(ai_response)
            else:
                # Try to get any AI message content as fallback
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        ai_response = msg.content
                        print(f"\nJARVIS: {ai_response}\n")
                        if tts:
                            tts.speak(ai_response)
                        break
                else:
                    logger.warning("⚠️ No AI response message found in result")
                    logger.info("Message types in result:")
                    for i, msg in enumerate(result["messages"][-5:]):
                        logger.info(f"  {i}: {type(msg).__name__} - Has content: {bool(getattr(msg, 'content', None))} - Has tool_calls: {bool(getattr(msg, 'tool_calls', None))}")
            
        except Exception as e:
            error_response = f"I apologise, Sir, but I've encountered an unexpected error: {str(e)}. Shall I attempt to diagnose the issue?"
            print(f"\nJARVIS: {error_response}\n")
            if tts:
                tts.speak(error_response)

if __name__ == "__main__":
    # First run a detailed connection test
    print("🔧 JARVIS System Initialization")
    print("=" * 60)
    
    # Try to run the main program
    asyncio.run(main())