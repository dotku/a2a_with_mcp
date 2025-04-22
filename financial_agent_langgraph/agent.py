"""
Financial Agent based on LangGraph for financial analysis.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import random
import os
import traceback
from uuid import uuid4
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END, MessagesState, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

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

# Dummy financial data (as a placeholder for PostgreSQL MCP server data)
DUMMY_FINANCIAL_DATA = {
    "companies": {
        "AAPL": {
            "name": "Apple Inc.",
            "sector": "Technology",
            "financial_metrics": {
                "revenue": [365.82, 394.33, 365.82],  # In billions USD, last 3 years
                "profit_margin": [0.25, 0.24, 0.23],
                "pe_ratio": 27.5,
                "price_to_book": 35.9,
                "dividend_yield": 0.51,
                "debt_to_equity": 1.9,
                "roi": 0.45,
                "eps_growth": 0.09,
                "cash_flow": 121.1,  # In billions USD
            },
            "stock_price": {
                "current": 175.50,
                "52_week_high": 198.23,
                "52_week_low": 143.90,
                "avg_volume": 60000000,
            },
            "historical_prices": {
                # Format: [timestamp, price]
                "daily": [[int((datetime.now() - timedelta(days=i)).timestamp()) * 1000, 
                          175.50 - random.uniform(-5, 5)] for i in range(30)],
                "weekly": [[int((datetime.now() - timedelta(weeks=i)).timestamp()) * 1000, 
                           175.50 - random.uniform(-10, 10)] for i in range(12)],
            }
        },
        "MSFT": {
            "name": "Microsoft Corporation",
            "sector": "Technology",
            "financial_metrics": {
                "revenue": [211.92, 198.27, 168.09],  # In billions USD, last 3 years
                "profit_margin": [0.36, 0.37, 0.34],
                "pe_ratio": 33.6,
                "price_to_book": 12.8,
                "dividend_yield": 0.72,
                "debt_to_equity": 0.42,
                "roi": 0.39,
                "eps_growth": 0.18,
                "cash_flow": 89.5,  # In billions USD
            },
            "stock_price": {
                "current": 407.48,
                "52_week_high": 434.90,
                "52_week_low": 309.98,
                "avg_volume": 26000000,
            },
            "historical_prices": {
                # Format: [timestamp, price]
                "daily": [[int((datetime.now() - timedelta(days=i)).timestamp()) * 1000, 
                          407.48 - random.uniform(-8, 8)] for i in range(30)],
                "weekly": [[int((datetime.now() - timedelta(weeks=i)).timestamp()) * 1000, 
                           407.48 - random.uniform(-15, 15)] for i in range(12)],
            }
        }
    },
    "market_indices": {
        "S&P500": {
            "current": 5021.84,
            "change": 0.57,
            "historical": [[int((datetime.now() - timedelta(days=i)).timestamp()) * 1000, 
                           5021.84 - random.uniform(-50, 50)] for i in range(30)]
        },
        "NASDAQ": {
            "current": 15990.66,
            "change": 1.12,
            "historical": [[int((datetime.now() - timedelta(days=i)).timestamp()) * 1000, 
                           15990.66 - random.uniform(-150, 150)] for i in range(30)]
        }
    }
}

# Define custom financial tools
@tool
def get_company_financial_metrics(ticker: str) -> Dict[str, Any]:
    """
    Get financial metrics for a company by ticker symbol.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        Company financial metrics
    """
    logger.info(f"Tool called: get_company_financial_metrics for {ticker}")
    ticker = ticker.upper()
    if ticker not in DUMMY_FINANCIAL_DATA["companies"]:
        logger.warning(f"Company with ticker {ticker} not found")
        return {"error": f"Company with ticker {ticker} not found."}
    
    result = DUMMY_FINANCIAL_DATA["companies"][ticker]["financial_metrics"]
    logger.info(f"Returning financial metrics for {ticker}")
    return result

@tool
def get_company_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Get current stock information for a company by ticker symbol.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        Current stock information
    """
    logger.info(f"Tool called: get_company_stock_info for {ticker}")
    ticker = ticker.upper()
    if ticker not in DUMMY_FINANCIAL_DATA["companies"]:
        logger.warning(f"Company with ticker {ticker} not found")
        return {"error": f"Company with ticker {ticker} not found."}
    
    result = DUMMY_FINANCIAL_DATA["companies"][ticker]["stock_price"]
    logger.info(f"Returning stock info for {ticker}: {result}")
    return result

@tool
def get_historical_prices(ticker: str, timeframe: str = "daily") -> List[List[float]]:
    """
    Get historical price data for a company.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL')
        timeframe: 'daily' or 'weekly'
        
    Returns:
        List of historical price data as [timestamp, price]
    """
    ticker = ticker.upper()
    if ticker not in DUMMY_FINANCIAL_DATA["companies"]:
        return {"error": f"Company with ticker {ticker} not found."}
    
    if timeframe not in ["daily", "weekly"]:
        return {"error": f"Invalid timeframe: {timeframe}. Use 'daily' or 'weekly'."}
    
    return DUMMY_FINANCIAL_DATA["companies"][ticker]["historical_prices"][timeframe]

@tool
def calculate_valuation_metrics(ticker: str) -> Dict[str, Any]:
    """
    Calculate key valuation metrics for a company.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        Dictionary of valuation metrics
    """
    ticker = ticker.upper()
    if ticker not in DUMMY_FINANCIAL_DATA["companies"]:
        return {"error": f"Company with ticker {ticker} not found."}
    
    company = DUMMY_FINANCIAL_DATA["companies"][ticker]
    metrics = company["financial_metrics"]
    stock = company["stock_price"]
    
    # Calculate some derived metrics
    result = {
        "intrinsic_value": round(stock["current"] * (1 + metrics["eps_growth"]) * 
                          (1 + metrics["profit_margin"]), 2),
        "price_to_sales": round(stock["current"] / (metrics["revenue"][-1] / 1e9), 2),
        "enterprise_value": round(stock["current"] * stock["avg_volume"] / 1e6 + 
                           metrics["debt_to_equity"] * metrics["cash_flow"], 2),
        "financial_health_score": round((metrics["profit_margin"][-1] * 0.3 + 
                                 (1 / metrics["debt_to_equity"]) * 0.3 +
                                 metrics["roi"] * 0.4) * 10, 1),
    }
    
    return result

@tool
def get_market_trends() -> Dict[str, Any]:
    """
    Get overall market trends and indices.
    
    Returns:
        Dictionary of market indices and trends
    """
    return DUMMY_FINANCIAL_DATA["market_indices"]

@tool
def analyze_financial_performance(ticker: str) -> Dict[str, Any]:
    """
    Perform comprehensive financial analysis on a company.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        Comprehensive financial analysis
    """
    ticker = ticker.upper()
    if ticker not in DUMMY_FINANCIAL_DATA["companies"]:
        return {"error": f"Company with ticker {ticker} not found."}
    
    company = DUMMY_FINANCIAL_DATA["companies"][ticker]
    metrics = company["financial_metrics"]
    
    # Calculate revenue growth
    rev = metrics["revenue"]
    revenue_growth = [(rev[i] - rev[i+1]) / rev[i+1] for i in range(len(rev)-1)]
    
    # Perform analysis
    analysis = {
        "name": company["name"],
        "sector": company["sector"],
        "summary": {
            "revenue_growth": [f"{g * 100:.2f}%" for g in revenue_growth],
            "profit_margin_trend": "stable" if abs(metrics["profit_margin"][0] - metrics["profit_margin"][-1]) < 0.03
                                   else "improving" if metrics["profit_margin"][0] > metrics["profit_margin"][-1]
                                   else "declining",
            "valuation": "high" if metrics["pe_ratio"] > 30 else "moderate" if metrics["pe_ratio"] > 15 else "low",
            "financial_health": "strong" if metrics["debt_to_equity"] < 1 else "moderate" if metrics["debt_to_equity"] < 2 else "concerning",
            "investment_rating": "buy" if metrics["roi"] > 0.4 and metrics["eps_growth"] > 0.1
                                else "hold" if metrics["roi"] > 0.3 or metrics["eps_growth"] > 0.05
                                else "sell"
        },
        "detailed_metrics": metrics,
    }
    
    return analysis

# Create the financial analysis tools
tools = [
    get_company_financial_metrics,
    get_company_stock_info,
    get_historical_prices,
    calculate_valuation_metrics,
    get_market_trends,
    analyze_financial_performance
]

# Create a ToolNode for executing tools
tool_node = ToolNode(tools)

# Initialize the LLM with the API key from environment
api_key = os.getenv("OPENAI_API_KEY")
logger.info(f"API key available: {api_key is not None}")

if api_key:
    # If we have an API key, use a real model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Bind tools to the LLM
    model_with_tools = llm.bind_tools(tools)
    logger.info("Using OpenAI model with tools")
else:
    # Fallback to a simple fake response mechanism to avoid API dependency
    logger.info("No API key found. Using fake response mode for development.")
    
    # Define a simple function to handle requests without needing an API
    def model_with_tools(messages):
        query = messages[-1].content if messages else ""
        logger.info(f"Received query in fake mode: {query}")
        
        # Check for common query patterns and return appropriate responses
        if "AAPL" in query or "Apple" in query:
            response = AIMessage(content="Based on my analysis of Apple (AAPL), the company has a P/E ratio of 27.5, with a profit margin of 25%. Their stock is currently trading at $175.50.")
            logger.info(f"Fake mode returning Apple response: {response.content}")
            return response
        elif "MSFT" in query or "Microsoft" in query:
            response = AIMessage(content="Microsoft (MSFT) shows strong financial health with a profit margin of 36% and PE ratio of 33.6. Their stock is currently trading at $407.48.")
            logger.info(f"Fake mode returning Microsoft response: {response.content}")
            return response
        elif "market" in query.lower() or "index" in query.lower():
            response = AIMessage(content="The current market trends show the S&P500 at 5021.84 and NASDAQ at 15990.66. Both indices have been showing positive momentum recently.")
            logger.info(f"Fake mode returning market response: {response.content}")
            return response
        elif "current" in query.lower() and ("stock" in query.lower() or "price" in query.lower()):
            if "AAPL" in query:
                response = AIMessage(content="The current stock price for Apple (AAPL) is $175.50.")
                logger.info(f"Fake mode returning Apple current price: {response.content}")
                return response
            elif "MSFT" in query:
                response = AIMessage(content="The current stock price for Microsoft (MSFT) is $407.48.")
                logger.info(f"Fake mode returning Microsoft current price: {response.content}")
                return response
        else:
            response = AIMessage(content="Based on the financial data available, I can provide insights into company performance, valuation metrics, and market trends. Please specify a company ticker or market index for more detailed analysis.")
            logger.info(f"Fake mode returning default response: {response.content}")
            return response

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
        
        # Add system prompt for financial analysis
        system_message = HumanMessage(content="""
        You are a financial analysis expert. Please analyze the request and use your financial tools to provide insights.
        Only use the tools available to you. Don't make up information.
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
                response = model_with_tools.invoke(messages)
            
            logger.info(f"Model response received: {response}")
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error calling model: {e}")
            logger.error(traceback.format_exc())
            # Return a fallback response
            error_msg = AIMessage(content=f"I encountered an error while analyzing your request: {str(e)}")
            return {"messages": [error_msg]}
    except Exception as e:
        logger.error(f"Unexpected error in call_model: {e}")
        logger.error(traceback.format_exc())
        # Return an empty response as fallback
        error_msg = AIMessage(content="I encountered an unexpected error while processing your request.")
        return {"messages": [error_msg]}

def process_financial_task(task: Task) -> Any:
    """
    Process a financial analysis task using LangGraph.
    
    Args:
        task: The A2A Task object
        
    Returns:
        Updated task with results
    """
    try:
        # Extract information from the task
        logger.info(f"Starting to process financial task: {task.id}")
        logger.debug(f"Full task details: {task}")
        
        state = extract_task_from_message(task)
        
        # Log the state for debugging
        logger.info(f"Extracted state: {state}")
        
        # Get the messages from the state
        messages = state["messages"]
        
        # Add system prompt for financial analysis
        system_message = HumanMessage(content="""
        You are a financial analysis expert. Please analyze the request and use your financial tools to provide insights.
        Only use the tools available to you. Don't make up information.
        """)
        
        # Add the system message at the beginning
        messages = [system_message] + messages
        logger.info(f"Prepared messages for model: {messages}")
        
        try:
            # Create a proper LangGraph workflow
            logger.info("Setting up LangGraph workflow")
            workflow = StateGraph(MessagesState)
            
            # Add nodes to the graph
            workflow.add_node("agent", call_model)
            workflow.add_node("tools", tool_node)
            
            # Add the START edge to define the entry point
            workflow.add_edge("__start__", "agent")
            
            # Add edges - create the conversational flow
            # workflow.add_edge("agent", "tools")
            workflow.add_edge("tools", "agent")
            
            # Set conditional routing to determine if we execute tools or finish
            workflow.add_conditional_edges(
                "agent",
                should_continue,
                {
                    "tools": "tools",
                    END: END
                }
            )
            
            # Compile the graph
            logger.info("Compiling LangGraph workflow")
            app = workflow.compile()
            
            # Execute the full conversation
            logger.info("Executing LangGraph conversation")
            result = app.invoke({"messages": messages}, config={"max_iterations": 50})
            logger.info(f"LangGraph execution complete: {result}")
            
            # Get the final message
            final_messages = result["messages"]
            logger.info(f"Final messages: {final_messages}")
            
            # We need to find the most informative AI message with actual content
            # Often after tool usage, earlier messages have more substance than the final pleasantries
            response_text = ""
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                    # Skip generic pleasantries like "If you have any other requests..."
                    if "If you have any other requests" not in msg.content and len(msg.content) > 30:
                        response_text = msg.content
                        logger.info(f"Found substantial response: {response_text[:100]}...")
                        break

            # If we didn't find a substantial message, fall back to the last message
            if not response_text and final_messages:
                last_message = final_messages[-1]
                if hasattr(last_message, 'content') and last_message.content:
                    response_text = last_message.content
                else:
                    response_text = str(last_message)

            logger.info(f"Final response text extracted: {response_text}")
            
        except Exception as model_error:
            logger.error(f"Error in LangGraph execution: {model_error}")
            logger.error(traceback.format_exc())
            # Create a fallback response
            response_text = f"Error analyzing financial data: {str(model_error)}"
        
        # Create agent message with the final response
        response_message = create_agent_message(response_text)
        logger.info(f"Created agent message: {response_message}")
        
        # Create updated task with the response
        logger.info("Creating updated task with response")
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
        logger.debug(f"Full updated task: {updated_task}")
        
        return updated_task
    except Exception as e:
        logger.error(f"Error in process_financial_task: {e}")
        logger.error(traceback.format_exc())
        # Create an error response
        error_message = create_agent_message(f"Error while processing financial task: {str(e)}")
        logger.info(f"Created error message for response: {error_message}")
        
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