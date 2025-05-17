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
import json

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

# Configure MCP servers
MCP_SERVER_CONFIG = {
    "postgres": {
        "command": "python3",
        "args": ["MCP-servers/postgres_mcp.py"],
        "transport": "stdio",
        "env": {}
    },
    "crypto": {
        "command": "npx",
        "args": ["mcp-crypto-price"],
        "transport": "stdio",
        "env": {
            "COINCAP_API_KEY": os.environ.get("COINCAP_API_KEY", "")
        }
    }
}

# Global variables for MCP state
mcp_client = None
all_fetched_tools: List[Any] = []
filtered_tools: List[Any] = []
tool_node = ToolNode([])
mcp_initialized = False
mcp_event_loop = None

# Create a global event loop for all MCP operations
def get_mcp_event_loop():
    """Get or create the MCP event loop."""
    global mcp_event_loop
    if mcp_event_loop is None or mcp_event_loop.is_closed():
        mcp_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(mcp_event_loop)
        logger.info("Created new MCP event loop")
    return mcp_event_loop

# Async function to initialize MCP client
async def init_mcp_client():
    """Initialize the MCP client asynchronously."""
    global mcp_client, all_fetched_tools, filtered_tools
    if mcp_client is not None:
        logger.info("MCP client already initialized")
        return filtered_tools
    
    try:
        logger.info("Attempting to initialize MCP client asynchronously...")
        if not os.environ.get("COINCAP_API_KEY"):
            logger.warning("COINCAP_API_KEY environment variable not set. Crypto analysis tools might have limitations or use CoinCap v2 API.")

        logger.debug("Instantiating MultiServerMCPClient...")
        mcp_client = MultiServerMCPClient(MCP_SERVER_CONFIG)
        logger.debug("MultiServerMCPClient instantiated. Calling __aenter__...")
        await mcp_client.__aenter__()
        logger.debug("MultiServerMCPClient.__aenter__ completed. Fetching tools...")
        all_fetched_tools = mcp_client.get_tools()
        logger.debug(f"MCP client get_tools() completed. Raw tools fetched: {[t.name for t in all_fetched_tools]}")

        logger.info(f"MCP client initialized with fetched tools: {[t.name for t in all_fetched_tools]}")

        allowed_crypto_tool_names = {"get-market-analysis", "get-historical-analysis"}
        filtered_tools = []
        for tool in all_fetched_tools:
            if tool.name in allowed_crypto_tool_names:
                filtered_tools.append(tool)
                logger.debug(f"Keeping crypto tool: {tool.name}")
            elif "crypto" not in tool.name.lower() and "price" not in tool.name.lower():
                if tool.name != 'get-crypto-price':
                    filtered_tools.append(tool)
                    logger.debug(f"Keeping non-crypto/non-price tool: {tool.name}")
            else:
                logger.debug(f"Filtering out tool: {tool.name}")

        logger.info(f"Final filtered tools available to agent: {[t.name for t in filtered_tools]}")
        return filtered_tools
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
            mcp_client = None
        except Exception as e:
            logger.error(f"Error shutting down MCP client: {e}")

# Safe synchronous cleanup function for atexit
def sync_cleanup_mcp():
    """Synchronous wrapper for cleanup to use with atexit."""
    global mcp_client, mcp_event_loop
    if mcp_client:
        try:
            loop = get_mcp_event_loop()
            if not loop.is_running():
                loop.run_until_complete(cleanup_mcp())
                logger.info("MCP client cleanup completed")
        except Exception as e:
            logger.error(f"Error in sync cleanup: {e}")

# Register the sync version for atexit
atexit.register(sync_cleanup_mcp)

api_key = os.getenv("OPENAI_API_KEY")
logger.info(f"API key available: {api_key is not None}")

if api_key:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model_with_tools = llm
    logger.info("Using OpenAI model")
else:
    logger.warning("No API key found. Using fake response mode for development.")
    def model_with_tools(messages):
        query = messages[-1].content if messages else ""
        logger.info(f"Received query in fake mode: {query}")
        response = AIMessage(content="I need to access the PostgreSQL database to provide financial data. Please ensure the database is properly configured and the MCP server is running.")
        logger.info(f"Fake mode returning database response: {response.content}")
        return response

# Ensure MCP client is initialized before processing tasks
async def ensure_mcp_initialized():
    """Ensure MCP client is initialized before processing tasks."""
    global mcp_initialized, model_with_tools, filtered_tools, tool_node
    if not mcp_initialized:
        logger.info("First-time MCP initialization check: MCP not yet initialized.")
        logger.debug("Calling init_mcp_client...")
        loaded_and_filtered_tools = await init_mcp_client()
        logger.debug(f"init_mcp_client returned: {[t.name for t in loaded_and_filtered_tools] if loaded_and_filtered_tools else 'None'}")
        if loaded_and_filtered_tools:
            filtered_tools = loaded_and_filtered_tools
            tool_node = ToolNode(filtered_tools)
            logger.info(f"ToolNode initialized with {len(filtered_tools)} tools.")
            if api_key:
                model_with_tools = llm.bind_tools(filtered_tools)
                logger.info(f"FILTERED Tools bound to model: {len(filtered_tools)} tools available")
        else:
            logger.warning("MCP tools not initialized or failed. Using empty ToolNode.")
            tool_node = ToolNode([])
        mcp_initialized = True
    return filtered_tools
        
from financial_agent_langgraph.common.types import Task, TextPart, TaskStatus, TaskState

def extract_task_from_message(task: Task) -> Dict[str, Any]:
    """Extract task information from a message."""
    history = task.history or []
    query = history[-1] if history else None
    if not query or query.role != "user":
        return {"messages": [], "task_id": task.id}
    text_content = []
    for part in query.parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text_content.append(part.get("text", ""))
        elif hasattr(part, "type") and part.type == "text":
            text_content.append(part.text)
    query_text = " ".join(text_content)
    return {"messages": [HumanMessage(content=query_text)], "task_id": task.id}

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
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info("Model requested tool use, continuing to tools node")
            return "tools"
        if hasattr(last_message, 'content') and last_message.content:
            logger.info(f"Model provided a final response: {last_message.content[:50]}...")
            return END
    logger.info("No clear next step determined, ending graph")
    return END

def call_model(state: MessagesState):
    """Call the LLM with the current messages."""
    try:
        messages = state["messages"]
        logger.info(f"Calling model with messages: {messages}")
        system_message = HumanMessage(content="""\
You are a financial analysis expert with access to a PostgreSQL database and external cryptocurrency market analysis tools.

Database Tools:
- Use the 'query' tool to run SQL queries against the database for internal financial data, including CURRENT crypto prices.
  Example: SELECT price_usd FROM crypto_quotes WHERE symbol = 'BTC' ORDER BY timestamp DESC LIMIT 1;

External Crypto Analysis Tools (Use these for market context and history, NOT current price):
- get-market-analysis: Use this tool to get detailed market analysis for a specific cryptocurrency, including top exchanges, volume distribution, and VWAP. To call this tool, provide the cryptocurrency symbol as an argument (e.g., for Bitcoin, the argument would be {\\"symbol\\": \\"BTC\\"}).
- get-historical-analysis: Use this tool to get historical price analysis for a specific cryptocurrency.
    To use this tool, provide the 'symbol' and optionally 'days' and 'interval'. For example:
    - To get 7 days of history for Solana (default interval): call with arguments {\\"symbol\\": \\"SOL\\", \\"days\\": 7}
    - To get 30 days of daily history for Bitcoin: call with arguments {\\"symbol\\": \\"BTC\\", \\"interval\\": \\"d1\\", \\"days\\": 30}
        
        Database schema:
- crypto_quotes (
    id SERIAL,
    symbol TEXT,
    price_usd DOUBLE PRECISION,
    market_cap DOUBLE PRECISION,
    volume_24h DOUBLE PRECISION,
    pct_change_1h DOUBLE PRECISION,
    pct_change_24h DOUBLE PRECISION,
    pct_change_7d DOUBLE PRECISION,
    timestamp TIMESTAMP
  )
- crypto_listings (
    id SERIAL,
    symbol TEXT,
    cmc_rank INTEGER,
    circulating_supply DOUBLE PRECISION,
    total_supply DOUBLE PRECISION,
    max_supply DOUBLE PRECISION,
    num_market_pairs INTEGER,
    volume_24h DOUBLE PRECISION,
    timestamp TIMESTAMP
  )
- global_metrics (
    id SERIAL,
    metric TEXT,
    value DOUBLE PRECISION,
    timestamp TIMESTAMP
  )
- price_conversions (
    id SERIAL,
    base_symbol TEXT,
    target_symbol TEXT,
    rate DOUBLE PRECISION,
    timestamp TIMESTAMP
  )
- id_map (symbol TEXT PRIMARY KEY, cmc_id INTEGER)
- metadata_info (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    logo_url TEXT,
    description TEXT,
    tags TEXT[],
    date_added DATE
  )

Combine information from the database and the external analysis tools as needed to answer user queries comprehensively. Prioritize the database 'query' tool for fetching the most recent price.
""")
        if not any(isinstance(msg, HumanMessage) and "financial analysis expert" in msg.content for msg in messages):
            messages = [system_message] + messages
        if callable(model_with_tools) and not hasattr(model_with_tools, 'invoke'):
            response = model_with_tools(messages)
        else:
            response = model_with_tools.invoke(messages)
        logger.info(f"Model response received: {response}")
        return {"messages": messages + [response]}
    except Exception as e:
        logger.error(f"Error calling model: {e}")
        logger.error(traceback.format_exc())
        error_content = f"Error during model execution: {e}"
        try:
            error_content = json.dumps({"error": str(e), "details": traceback.format_exc()})
        except TypeError:
            pass # Keep the basic error string if JSON serialization fails
        return {"messages": [AIMessage(content=f"Sorry, an internal error occurred: {error_content}")]}

async def process_financial_task_async(task: Task) -> Any:
    """
    Process a financial analysis task using LangGraph asynchronously.
    """
    try:
        await ensure_mcp_initialized()
        state = extract_task_from_message(task)
        messages = state["messages"]
        system_message = HumanMessage(content="""
You are a financial analysis expert with access to a PostgreSQL database and external cryptocurrency market analysis tools.

Database Tools:
- Use the 'query' tool to run SQL queries against the database for internal financial data, including CURRENT crypto prices.
  Example: SELECT price_usd FROM crypto_quotes WHERE symbol = 'BTC' ORDER BY timestamp DESC LIMIT 1;

External Crypto Analysis Tools (Use these for market context and history, NOT current price):
- get-market-analysis: Use this tool to get detailed market analysis for a specific cryptocurrency, including top exchanges, volume distribution, and VWAP. To call this tool, provide the cryptocurrency symbol as an argument (e.g., for Bitcoin, the argument would be {\\"symbol\\": \\"BTC\\"}).
- get-historical-analysis: Use this tool to get historical price analysis for a specific cryptocurrency.
    To use this tool, provide the 'symbol' and optionally 'days' and 'interval'. For example:
    - To get 7 days of history for Solana (default interval): call with arguments {\\"symbol\\": \\"SOL\\", \\"days\\": 7}
    - To get 30 days of daily history for Bitcoin: call with arguments {\\"symbol\\": \\"BTC\\", \\"interval\\": \\"d1\\", \\"days\\": 30}
        
        Database schema:
- crypto_quotes (
    id SERIAL,
    symbol TEXT,
    price_usd DOUBLE PRECISION,
    market_cap DOUBLE PRECISION,
    volume_24h DOUBLE PRECISION,
    pct_change_1h DOUBLE PRECISION,
    pct_change_24h DOUBLE PRECISION,
    pct_change_7d DOUBLE PRECISION,
    timestamp TIMESTAMP
  )
- crypto_listings (
    id SERIAL,
    symbol TEXT,
    cmc_rank INTEGER,
    circulating_supply DOUBLE PRECISION,
    total_supply DOUBLE PRECISION,
    max_supply DOUBLE PRECISION,
    num_market_pairs INTEGER,
    volume_24h DOUBLE PRECISION,
    timestamp TIMESTAMP
  )
- global_metrics (
    id SERIAL,
    metric TEXT,
    value DOUBLE PRECISION,
    timestamp TIMESTAMP
  )
- price_conversions (
    id SERIAL,
    base_symbol TEXT,
    target_symbol TEXT,
    rate DOUBLE PRECISION,
    timestamp TIMESTAMP
  )
- id_map (symbol TEXT PRIMARY KEY, cmc_id INTEGER)
- metadata_info (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    logo_url TEXT,
    description TEXT,
    tags TEXT[],
    date_added DATE
  )

Combine information from the database and the external analysis tools as needed to answer user queries comprehensively. Prioritize the database 'query' tool for fetching the most recent price.
        """)
        messages = [system_message] + messages
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_edge("tools", "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", END: END}
        )
        app = workflow.compile()
        result = await app.ainvoke({"messages": messages}, config={"max_iterations": 10})
        final_messages = result["messages"]
        response_text = ""
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                response_text = msg.content
                break
        if not response_text:
            last = final_messages[-1]
            response_text = last.content if hasattr(last, 'content') else str(last)

        # Create the final Task object on success
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
            history=(task.history or []) + [response_message],
            metadata=task.metadata
        )
        return updated_task

    except Exception as e:
        logger.error(f"Error in process_financial_task_async: {e}")
        logger.error(traceback.format_exc())
        response_text = f"Error while processing financial task: {str(e)}"
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
            history=(task.history or []) + [response_message],
            metadata=task.metadata
        )
        return updated_task

# Process a financial task using the shared MCP event loop
def process_financial_task(task: Task) -> Any:
    """Synchronous wrapper for process_financial_task_async."""
    loop = get_mcp_event_loop()
    try:
        return loop.run_until_complete(process_financial_task_async(task))
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(process_financial_task_async(task))
            raise 
