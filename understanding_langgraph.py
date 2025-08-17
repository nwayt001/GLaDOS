# Understanding ChatOllama and Prompts

## Method 1: Direct Message Passing (Similar to OpenAI)
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOllama(model="llama2", base_url="http://localhost:11434")

# You can pass messages directly, just like OpenAI
messages = [
    SystemMessage(content="You are GLaDOS..."),
    HumanMessage(content="Hello"),
    AIMessage(content="Oh, it's you..."),
    HumanMessage(content="How are you?")
]

response = llm.invoke(messages)
print(response.content)

## Method 2: Using Prompt Templates (What we're doing)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# This creates a reusable template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are GLaDOS..."),  # Static system message
    MessagesPlaceholder(variable_name="chat_history"),  # Dynamic history
    ("human", "{input}")  # Current user input
])

# When you use it:
chain = prompt | llm
response = chain.invoke({
    "chat_history": [  # Previous messages
        HumanMessage(content="Hello"),
        AIMessage(content="Oh, it's you...")
    ],
    "input": "How are you?"  # Current input
})

# The prompt template automatically constructs the full message list

## Method 3: Simple Chain Without History
simple_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are GLaDOS..."),
    ("human", "{input}")
])

simple_chain = simple_prompt | llm
response = simple_chain.invoke({"input": "Hello"})

# Understanding LangGraph Memory and State

## Traditional Approach (What you're used to):
class TraditionalChatbot:
    def __init__(self):
        self.messages = [{"role": "system", "content": "You are GLaDOS..."}]
    
    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        
        # Call API with full history
        response = openai_api_call(self.messages)
        
        self.messages.append({"role": "assistant", "content": response})
        return response

## LangGraph Approach:
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
import operator

# 1. Define State Structure
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    # The operator.add means new messages are APPENDED to the list

# 2. Define Node Function
def chat_node(state: AgentState) -> dict:
    # State contains all previous messages
    all_messages = state["messages"]
    
    # Call LLM with full history
    response = llm.invoke(all_messages)
    
    # Return ONLY the new message to append
    return {"messages": [response]}

# 3. Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)
app = workflow.compile()

# 4. Usage - State is managed by LangGraph
state = {"messages": [SystemMessage(content="You are GLaDOS...")]}

# First interaction
state["messages"].append(HumanMessage(content="Hello"))
state = app.invoke(state)  # State now has AI response appended

# Second interaction
state["messages"].append(HumanMessage(content="How are you?"))
state = app.invoke(state)  # State has full history

## How Memory Works in LangGraph:

# The key is the Annotated type with operator.add:
messages: Annotated[list[BaseMessage], operator.add]

# This tells LangGraph:
# 1. When a node returns {"messages": [new_message]}
# 2. Don't REPLACE state["messages"], but APPEND to it
# 3. It's like doing: state["messages"] += returned["messages"]

# So the flow is:
# Initial: state = {"messages": [SystemMessage, HumanMessage]}
# Node returns: {"messages": [AIMessage]}
# New state: {"messages": [SystemMessage, HumanMessage, AIMessage]}

# You can also use other operators:
# operator.add - Append/concatenate (for lists)
# operator.or_ - Merge (for dicts)
# Custom reducers for complex logic

## Comparison Summary:

# Traditional:
messages = []
messages.append(user_msg)
messages.append(ai_response)
# You manually manage the list

# LangGraph:
# Define state structure with reducer (operator.add)
# Nodes return partial updates
# LangGraph handles the state merging
# More complex workflows possible (branching, cycles, conditions)

## Why LangGraph?
# 1. Complex workflows - multiple agents, branching logic
# 2. State management - automatic handling of updates
# 3. Visualization - can see your workflow as a graph
# 4. Checkpointing - can save/resume conversations
# 5. Streaming - built-in support for streaming tokens
# 6. Parallelism - can run multiple nodes concurrently