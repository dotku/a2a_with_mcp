"""
Financial Agent based on LangGraph for financial analysis.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import traceback
import asyncio
from uuid import uuid4
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END, MessagesState, START
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
import atexit
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables before importing any models
load_dotenv()

from financial_agent_langgraph.common.types import (
    Part, 
    TextPart, 
    Task, 
    TaskState, 
    TaskStatus, 
    Message,
    Artifact
)

# Configure more verbose logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a file handler for persistent logging
try:
    file_handler = logging.FileHandler('financial_agent.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info("File logging initialized")
except Exception as e:
    logger.error(f"Failed to set up file logging: {e}")

# Configure the Postgres MCP server
MCP_SERVER_CONFIG = {
    "postgres": {
        "command": "python3",
        "args": ["MCP-servers/postgres_mcp.py"],
        "transport": "stdio",
    }
}

# Initialize MCP client and tools
mcp_client = None
tools = []

# Initialize tool_node, it will be updated later
tool_node = ToolNode([]) # Start with an empty list, will be updated

# Async function to initialize MCP client
async def init_mcp_client():
    """Initialize the MCP client asynchronously."""
    global mcp_client, tools
    
    try:
        logger.info("Initializing MCP client asynchronously...")
        mcp_client = MultiServerMCPClient(MCP_SERVER_CONFIG)
        await mcp_client.__aenter__()
        tools = mcp_client.get_tools()
        logger.info(f"MCP client initialized with tools: {[t.name for t in tools]}")
        return tools
    except Exception as e:
        logger.error(f"Failed to initialize MCP client: {e}")
        logger.error(traceback.format_exc())
        return []

# Function to clean up MCP client on exit
async def cleanup_mcp():
    """Clean up MCP client asynchronously."""
    global mcp_client
    if mcp_client:
        try:
            await mcp_client.__aexit__(None, None, None)
            logger.info("MCP client shut down")
        except Exception as e:
            logger.error(f"Error shutting down MCP client: {e}")

atexit.register(lambda: asyncio.create_task(cleanup_mcp()))

api_key = os.getenv("OPENAI_API_KEY")
logger.info(f"API key available: {api_key is not None}")

if api_key:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model_with_tools = llm # Bind tools later after initialization
    logger.info("Using OpenAI model")
else:
    logger.warning("No API key found. Using fake response mode for development.")
    def model_with_tools(messages):
        query = messages[-1].content if messages else ""
        logger.info(f"Received query in fake mode: {query}")
        response = AIMessage(content="I need to access the PostgreSQL database to provide financial data. Please ensure the database is properly configured and the MCP server is running.")
        logger.info(f"Fake mode returning database response: {response.content}")
        return response

mcp_initialized = False

async def ensure_mcp_initialized():
    """Ensure MCP client is initialized before processing tasks."""
    global mcp_initialized, model_with_tools, tools, tool_node
    
    if not mcp_initialized:
        logger.info("First-time MCP initialization")
        loaded_tools = await init_mcp_client()
        
        if loaded_tools:
            tools = loaded_tools # Update the global tools list
            tool_node = ToolNode(tools) # Create the REAL ToolNode with loaded tools
            logger.info(f"ToolNode initialized with {len(tools)} tools.")
            if api_key:
                model_with_tools = llm.bind_tools(tools) 
                logger.info(f"Tools bound to model: {len(tools)} tools available")
        else:
            logger.warning("MCP tools not initialized. Using empty ToolNode.")
            tool_node = ToolNode([]) # Ensure tool_node is a ToolNode even if empty
            
        mcp_initialized = True
        
    # Return the globally updated tools list
    return tools

def extract_task_from_message(task: Task) -> Dict[str, Any]:
    """Extract task information from a message."""
    history = task.history or []
    query = history[-1] if history else None
    
    if not query or query.role != "user":
        return {"messages": [], "task_id": task.id}
    
    # Extract text from parts
    text_content = []
    for part in query.parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text_content.append(part.get("text", ""))
        elif hasattr(part, "type") and part.type == "text":
            text_content.append(part.text)
    
    query_text = " ".join(text_content)
    
    # Create a standardized state with proper MessagesState format
    return {
        "messages": [HumanMessage(content=query_text)],
        "task_id": task.id
    }

def create_agent_message(content: str) -> Message:
    """Create an agent message with text content."""
    return Message(
        role="agent",
        parts=[TextPart(text=content)]
    )

def should_continue(state: MessagesState):
    """Determine if we should continue with tool execution or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the message is from the AI
    if isinstance(last_message, AIMessage):
        # If the message has tool_calls, continue to the tools node
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info("Model requested tool use, continuing to tools node")
            return "tools"
        
        # If the message has actual text content, we're done
        if hasattr(last_message, 'content') and last_message.content:
            logger.info(f"Model provided a final response: {last_message.content[:50]}...")
            return END
    
    # Default case - if nothing else matched, end the graph
    logger.info("No clear next step determined, ending graph")
    return END

def call_model(state: MessagesState):
    """Call the LLM with the current messages."""
    try:
        messages = state["messages"]
        logger.info(f"Calling model with messages: {messages}")
        
        # Add system prompt for financial analysis with Postgres
        system_message = HumanMessage(content="""
        You are a financial analysis expert with access to a PostgreSQL database. 
        You can use the following tools to help with your analysis:
        
        - query: Run SQL queries against the database to get financial data
          Example: SELECT * FROM stocks WHERE symbol = 'AAPL';
          
        - fetch_financial_snapshot: Get the latest financial snapshot data
          This returns the latest consolidated financial data as a JSON object
        
        Database schema:
        - stocks (symbol TEXT, price NUMERIC, volume INTEGER, timestamp TIMESTAMP)
        - financial_snapshot (id SERIAL, data JSONB, created_at TIMESTAMP)
        - companies (ticker TEXT, name TEXT, sector TEXT, industry TEXT)
        - financial_metrics (ticker TEXT, metric TEXT, value NUMERIC, period TEXT)
        
        Always verify the data available in the database before making claims.
        When using the 'query' tool, make sure to write proper SQL queries.
        Provide clear insights based on the data returned from the database.
        """)
        
        # Add the system message at the beginning if not already there
        if not any(isinstance(msg, HumanMessage) and "financial analysis expert" in msg.content for msg in messages):
            messages = [system_message] + messages
            logger.info("Added system message to prompt")
        
        # Use the correct invocation method based on the type of model_with_tools
        try:
            if callable(model_with_tools) and not hasattr(model_with_tools, 'invoke'):
                # For the fake function version
                logger.info("Using direct function call for model")
                response = model_with_tools(messages)
            else:
                # For the RunnableBinding version
                logger.info("Using .invoke() method for model")
                response = model_with_tools.invoke(messages) # Note: This might need to be async if model_with_tools is async
            
            logger.info(f"Model response received: {response}")
            return {"messages": messages + [response]}
        except Exception as e:
            logger.error(f"Error calling model: {e}")
            logger.error(traceback.format_exc())
            # Return a fallback response
            error_msg = AIMessage(content=f"I encountered an error while analyzing your request: {str(e)}")
            return {"messages": messages + [error_msg]}
    except Exception as e:
        logger.error(f"Unexpected error in call_model: {e}")
        logger.error(traceback.format_exc())
        # Return an empty response as fallback
        error_msg = AIMessage(content="I encountered an unexpected error while processing your request.")
        return {"messages": messages + [error_msg]}

async def process_financial_task_async(task: Task) -> Any:
    """
    Process a financial analysis task using LangGraph asynchronously.
    
    Args:
        task: The A2A Task object
        
    Returns:
        Updated task with results
    """
    try:
        # Ensure MCP is initialized and ToolNode is updated
        await ensure_mcp_initialized()
        
        # Extract information from the task
        logger.info(f"Starting to process financial task: {task.id}")
        logger.debug(f"Full task details: {task}")
        
        state = extract_task_from_message(task)
        logger.info(f"Extracted state: {state}")
        
        messages = state["messages"]
        system_message = HumanMessage(content="""
        You are a financial analysis expert with access to a PostgreSQL database. 
        You can use the following tools to help with your analysis:
        
        - query: Run SQL queries against the database to get financial data
          Example: SELECT * FROM stocks WHERE symbol = 'AAPL';
          
        - fetch_financial_snapshot: Get the latest financial snapshot data
          This returns the latest consolidated financial data as a JSON object
        
        Database schema:
        - stocks (symbol TEXT, price NUMERIC, volume INTEGER, timestamp TIMESTAMP)
        - financial_snapshot (id SERIAL, data JSONB, created_at TIMESTAMP)
        - companies (ticker TEXT, name TEXT, sector TEXT, industry TEXT)
        - financial_metrics (ticker TEXT, metric TEXT, value NUMERIC, period TEXT)
        
        Always verify the data available in the database before making claims.
        When using the 'query' tool, make sure to write proper SQL queries.
        Provide clear insights based on the data returned from the database.
        
        IMPORTANT: Remember to use async invocation for tools - do not use sync methods.
        """)
        messages = [system_message] + messages
        logger.info(f"Prepared messages for model: {messages}")
        
        try:
            logger.info("Setting up LangGraph workflow")
            workflow = StateGraph(MessagesState)
            
            # Add nodes - agent and the CORRECT ToolNode
            workflow.add_node("agent", call_model)
            workflow.add_node("tools", tool_node) # Use the globally updated tool_node
            logger.info("Added agent and standard ToolNode to workflow")

            workflow.add_edge(START, "agent")
            workflow.add_edge("tools", "agent")
            workflow.add_conditional_edges(
                "agent",
                should_continue,
                {"tools": "tools", END: END}
            )
            
            logger.info("Compiling LangGraph workflow")
            app = workflow.compile()
            
            logger.info("Executing LangGraph conversation asynchronously")
            # CRITICAL: Use await app.ainvoke
            result = await app.ainvoke({"messages": messages}, config={"max_iterations": 50}) 
            logger.info(f"LangGraph execution complete: {result}")
            
            final_messages = result["messages"]
            logger.info(f"Final messages: {final_messages}")
            
            response_text = ""
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                    if "If you have any other requests" not in msg.content and len(msg.content) > 30:
                        response_text = msg.content
                        logger.info(f"Found substantial response: {response_text[:100]}...")
                        break
            if not response_text and final_messages:
                last_message = final_messages[-1]
                if hasattr(last_message, 'content') and last_message.content:
                    response_text = last_message.content
                else:
                    response_text = str(last_message)
            logger.info(f"Final response: {response_text}")

        except Exception as model_error:
            logger.error(f"Error in LangGraph execution: {model_error}")
            logger.error(traceback.format_exc())
            response_text = f"Error analyzing financial data: {str(model_error)}"
        
        response_message = create_agent_message(response_text)
        updated_task = Task(
            id=task.id,
            sessionId=task.sessionId,
            status=TaskStatus(
                state=TaskState.COMPLETED,
                message=response_message,
                timestamp=datetime.now()
            ),
            artifacts=[
                Artifact(
                    name="financial_insights",
                    description="Financial analysis results",
                    parts=[TextPart(text=response_text)]
                )
            ],
            history=task.history + [response_message] if task.history else [response_message],
            metadata=task.metadata
        )
        logger.info(f"Updated task created: {updated_task.id}, status: {updated_task.status.state}")
        return updated_task

    except Exception as e:
        logger.error(f"Error in process_financial_task_async: {e}")
        logger.error(traceback.format_exc())
        error_message = create_agent_message(f"Error while processing financial task: {str(e)}")
        error_task = Task(
            id=task.id,
            sessionId=task.sessionId,
            status=TaskStatus(
                state=TaskState.FAILED,
                message=error_message,
                timestamp=datetime.now()
            ),
            history=task.history + [error_message] if task.history else [error_message],
            metadata=task.metadata
        )
        logger.info(f"Error task created: {error_task.id}, status: {error_task.status.state}")
        return error_task

# Synchronous wrapper remains the same
def process_financial_task(task: Task) -> Any:
    """Synchronous wrapper for process_financial_task_async."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(process_financial_task_async(task))
    finally:
        loop.close() 