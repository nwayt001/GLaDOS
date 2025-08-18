import os
import asyncio
import requests
from typing import TypedDict, Annotated, Sequence, List, Optional, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
# We'll implement our own search instead of using the deprecated package
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator
import pygame
import io
import threading
import logging
import json
from datetime import datetime
import subprocess
import shlex
from pathlib import Path
import re

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

@tool
def get_weather(location: str) -> str:
    """Get current weather and forecast for a location. Returns current conditions and next 2 days forecast."""
    try:
        # Use wttr.in which provides both current and forecast data for free
        print(f"   🌤️ Checking weather for {location}...")
        url = f"https://wttr.in/{location}?format=j1"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Current weather
            current = data['current_condition'][0]
            temp_c = current['temp_C']
            temp_f = current['temp_F']
            feels_c = current['FeelsLikeC']
            feels_f = current['FeelsLikeF']
            description = current['weatherDesc'][0]['value']
            humidity = current['humidity']
            wind_mph = current['windspeedMiles']
            wind_kph = current['windspeedKmph']
            
            result = [f"Current weather in {location}:"]
            result.append(f"Temperature: {temp_f}°F ({temp_c}°C), feels like {feels_f}°F ({feels_c}°C)")
            result.append(f"Conditions: {description}")
            result.append(f"Humidity: {humidity}%, Wind: {wind_mph} mph ({wind_kph} km/h)")
            
            # Forecast for next 2 days
            if 'weather' in data:
                result.append("\nForecast:")
                for i, day in enumerate(data['weather'][:3]):  # Today + next 2 days
                    date = day['date']
                    max_temp_f = day['maxtempF']
                    max_temp_c = day['maxtempC']
                    min_temp_f = day['mintempF']
                    min_temp_c = day['mintempC']
                    
                    # Get hourly data for evening (around 6 PM)
                    evening_desc = "No evening data"
                    for hour in day['hourly']:
                        if hour['time'] == '1800':  # 6 PM
                            evening_desc = hour['weatherDesc'][0]['value']
                            evening_temp_f = hour['tempF']
                            evening_temp_c = hour['tempC']
                            evening_desc = f"{evening_desc}, {evening_temp_f}°F ({evening_temp_c}°C)"
                            break
                    
                    if i == 0:
                        result.append(f"• Today ({date}): High {max_temp_f}°F ({max_temp_c}°C), Low {min_temp_f}°F ({min_temp_c}°C)")
                        result.append(f"  Evening: {evening_desc}")
                    elif i == 1:
                        result.append(f"• Tomorrow ({date}): High {max_temp_f}°F ({max_temp_c}°C), Low {min_temp_f}°F ({min_temp_c}°C)")
                        result.append(f"  Evening: {evening_desc}")
                    else:
                        day_name = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][
                            datetime.strptime(date, "%Y-%m-%d").weekday()
                        ]
                        result.append(f"• {day_name} ({date}): High {max_temp_f}°F ({max_temp_c}°C), Low {min_temp_f}°F ({min_temp_c}°C)")
            
            return "\n".join(result)
        else:
            return f"I couldn't retrieve weather data for {location}. Please check the city name, Sir."
            
    except Exception as e:
        return f"I apologise, Sir, but I encountered an error checking the weather: {str(e)}"

@tool
def execute_bash(command: str) -> str:
    """Execute a bash command and return the output. Use with caution."""
    print(f"   ⚡ Executing: {command}")
    try:
        # Safety check - warn about potentially dangerous commands
        dangerous_patterns = ['rm -rf /', 'dd if=', 'mkfs', ':(){ :|:& };:']
        if any(pattern in command.lower() for pattern in dangerous_patterns):
            return "I must decline to execute this potentially dangerous command, Sir. Perhaps we should reconsider our approach."
        
        # Execute the command with timeout
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd()
        )
        
        output = result.stdout if result.stdout else result.stderr
        if not output and result.returncode == 0:
            output = "Command executed successfully with no output."
        elif not output:
            output = f"Command failed with return code {result.returncode}"
            
        # Truncate very long outputs
        if len(output) > 2000:
            output = output[:2000] + "\n... (output truncated)"
            
        return output
    except subprocess.TimeoutExpired:
        return "The command timed out after 30 seconds, Sir. It may still be running in the background."
    except Exception as e:
        return f"I encountered an error executing the command: {str(e)}"

@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    print(f"   📖 Reading: {file_path}")
    try:
        path = Path(file_path).expanduser().resolve()
        
        # Safety check - don't read sensitive files
        sensitive_patterns = ['.ssh/', '.aws/', '.env', 'password', 'secret', 'token', 'key']
        if any(pattern in str(path).lower() for pattern in sensitive_patterns):
            return "I must advise against reading potentially sensitive files, Sir. Security protocols prevent me from accessing this file."
        
        if not path.exists():
            return f"The file {file_path} does not exist, Sir."
        
        if not path.is_file():
            return f"{file_path} is not a file, Sir."
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > 1_000_000:  # 1MB limit
            return f"The file is quite large ({file_size:,} bytes), Sir. Perhaps we should use a different approach for such large files."
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Truncate if still too long
        if len(content) > 5000:
            content = content[:5000] + "\n... (content truncated)"
            
        return content
    except UnicodeDecodeError:
        return "The file appears to be binary or uses an unsupported encoding, Sir."
    except Exception as e:
        return f"I encountered an error reading the file: {str(e)}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist."""
    print(f"   ✍️ Writing to: {file_path}")
    try:
        path = Path(file_path).expanduser().resolve()
        
        # Safety check - don't overwrite system files
        system_dirs = ['/etc', '/usr', '/bin', '/sbin', '/boot', '/dev', '/proc', '/sys']
        if any(str(path).startswith(d) for d in system_dirs):
            return "I cannot modify system files, Sir. This would be inadvisable."
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup existing file if it exists
        if path.exists():
            backup_path = path.with_suffix(path.suffix + '.backup')
            print(f"   💾 Creating backup: {backup_path}")
            path.rename(backup_path)
            
        # Write the new content
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"Successfully wrote {len(content)} characters to {file_path}, Sir."
    except Exception as e:
        return f"I encountered an error writing to the file: {str(e)}"

@tool
def list_directory(directory: str = ".") -> str:
    """List contents of a directory."""
    print(f"   📁 Listing: {directory}")
    try:
        path = Path(directory).expanduser().resolve()
        
        if not path.exists():
            return f"The directory {directory} does not exist, Sir."
        
        if not path.is_dir():
            return f"{directory} is not a directory, Sir."
        
        items = []
        for item in sorted(path.iterdir()):
            if item.is_dir():
                items.append(f"📁 {item.name}/")
            elif item.is_file():
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size/(1024*1024):.1f}MB"
                items.append(f"📄 {item.name} ({size_str})")
        
        if not items:
            return "The directory is empty, Sir."
        
        return "\n".join(items[:50])  # Limit to 50 items
    except Exception as e:
        return f"I encountered an error listing the directory: {str(e)}"

@tool
def check_web_scraper_status() -> str:
    """Check the status of the iOS app web scraper running on the Linux box."""
    print(f"   🔍 Checking web scraper status on Linux box...")
    
    linux_host = "nicholas@10.0.0.108"
    status_report = []
    
    try:
        # Check if we can connect to the Linux box
        print(f"   📡 Connecting to {linux_host}...")
        ping_cmd = f"ssh -o ConnectTimeout=5 -o BatchMode=yes {linux_host} 'echo Connected'"
        ping_result = subprocess.run(ping_cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        if ping_result.returncode != 0:
            return f"Cannot connect to the Linux box at {linux_host}. The system appears to be offline or SSH is not configured, Sir."
        
        # Check for the Red-Dot-Scraper process first
        print(f"   🔎 Searching for Red-Dot-Scraper process...")
        red_dot_cmd = f"ssh {linux_host} 'ps aux | grep \"Red-Dot-Scraper\" | grep -v grep'"
        red_dot_result = subprocess.run(red_dot_cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        processes_found = []
        if red_dot_result.stdout.strip():
            processes_found.append(("Red-Dot-Scraper", red_dot_result.stdout.strip()))
        
        # Also check for other common scraper process names
        scraper_patterns = ["scraper", "scrapy", "crawler", "spider", "selenium", "puppeteer", "playwright"]
        for pattern in scraper_patterns:
            cmd = f"ssh {linux_host} 'ps aux | grep -i {pattern} | grep -v grep'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.stdout.strip():
                processes_found.append((pattern, result.stdout.strip()))
        
        if processes_found:
            # Check if Red-Dot-Scraper specifically was found
            red_dot_found = any(pattern == "Red-Dot-Scraper" for pattern, _ in processes_found)
            if red_dot_found:
                status_report.append("✅ Red-Dot-Scraper is running!")
            else:
                status_report.append("✅ Web scraper processes detected:")
            
            for pattern, process in processes_found:
                # Parse process info for cleaner display
                lines = process.split('\n')
                for line in lines[:3]:  # Limit to first 3 matches per pattern
                    parts = line.split()
                    if len(parts) >= 11:
                        user = parts[0]
                        pid = parts[1]
                        cpu = parts[2]
                        mem = parts[3]
                        cmd = ' '.join(parts[10:])[:100]  # Truncate long commands
                        if pattern == "Red-Dot-Scraper":
                            status_report.append(f"   • 🔴 PID {pid}: {cmd} (CPU: {cpu}%, MEM: {mem}%)")
                        else:
                            status_report.append(f"   • PID {pid}: {cmd} (CPU: {cpu}%, MEM: {mem}%)")
        else:
            status_report.append("⚠️ Red-Dot-Scraper is NOT running!")
        
        # Check for Docker containers that might be running scrapers
        print(f"   🐳 Checking Docker containers...")
        docker_cmd = f"ssh {linux_host} 'docker ps --format \"table {{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.Ports}}}}\" 2>/dev/null | grep -E \"scraper|crawler|spider\" || true'"
        docker_result = subprocess.run(docker_cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        if docker_result.stdout.strip():
            status_report.append("\n📦 Docker containers:")
            status_report.append(docker_result.stdout.strip())
        
        # Check system resources
        print(f"   💻 Checking system resources...")
        resource_cmd = f"ssh {linux_host} 'echo \"=== System Resources ===\"; uptime; echo \"\"; free -h | head -2; echo \"\"; df -h / | tail -1'"
        resource_result = subprocess.run(resource_cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        if resource_result.stdout:
            status_report.append("\n💻 System Status:")
            for line in resource_result.stdout.split('\n'):
                if 'load average' in line:
                    # Extract load average
                    load_part = line.split('load average:')[1].strip() if 'load average:' in line else ''
                    status_report.append(f"   • Load Average: {load_part}")
                elif 'Mem:' in line:
                    # Parse memory info
                    parts = line.split()
                    if len(parts) >= 3:
                        status_report.append(f"   • Memory: {parts[1]} total, {parts[2]} used")
                elif '/' in line and '%' in line:
                    # Parse disk usage
                    parts = line.split()
                    if len(parts) >= 5:
                        status_report.append(f"   • Disk Usage: {parts[4]} used")
        
        # Check for recent scraper logs
        print(f"   📝 Checking for recent logs...")
        log_locations = [
            "/var/log/scraper.log",
            "~/scraper/logs/scraper.log",
            "~/logs/scraper.log",
            "/home/nicholas/scraper.log"
        ]
        
        for log_path in log_locations:
            log_cmd = f"ssh {linux_host} 'if [ -f {log_path} ]; then echo \"Found: {log_path}\"; tail -n 5 {log_path}; fi'"
            log_result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True, timeout=10)
            if log_result.stdout.strip():
                status_report.append(f"\n📋 Recent log entries from {log_path}:")
                status_report.append(log_result.stdout.strip()[:500])  # Limit log output
                break
        
        if not status_report:
            status_report.append("No specific scraper information found. You may need to check the specific service or process name, Sir.")
        
        return "\n".join(status_report)
        
    except subprocess.TimeoutExpired:
        return "The connection to the Linux box timed out, Sir. The system may be under heavy load or experiencing network issues."
    except Exception as e:
        return f"I encountered an error checking the scraper status: {str(e)}"

# Collect all tools
tools = [web_search, get_weather, execute_bash, read_file, write_file, list_directory, check_web_scraper_status]



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