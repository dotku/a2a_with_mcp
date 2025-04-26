"""Task manager for the sentiment analysis agent.

This file serves as an entry point for the UI to connect to our sentiment analysis agent.
It exposes the necessary HTTP endpoints for the A2A UI to register and communicate with the agent.
"""

import asyncio
import logging
import os
import sys
from fastapi import FastAPI, APIRouter
import uvicorn
from .agent import SentimentAnalysisAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_agent_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app and router
app = FastAPI()
router = APIRouter()
app.include_router(router)

# Initialize the sentiment analysis agent
sentiment_agent = SentimentAnalysisAgent()

@router.post("/invoke")
async def invoke_agent(request: dict):
    """Invoke the sentiment analysis agent with the given query."""
    try:
        query = request.get("query", "")
        session_id = request.get("session_id", "default_session")
        
        logger.info(f"Received request to invoke agent with query: {query}, session_id: {session_id}")
        
        # Call the synchronous agent
        response = sentiment_agent.invoke(query, session_id)
        
        # Extract the raw string response from CrewOutput
        if hasattr(response, 'raw'):
            response_text = response.raw
        else:
            response_text = str(response)
            
        logger.info(f"Agent response (first 100 chars): {response_text[:100] if response_text else 'None'}")
        
        return {"response": response_text}
    except Exception as e:
        logger.exception(f"Error invoking agent: {e}")
        return {"error": str(e)}

@router.get("/info")
async def agent_info():
    """Return information about this agent for the UI to display."""
    return {
        "name": "Bitcoin Sentiment Analyst",
        "description": "Analyzes Reddit data to determine Bitcoin sentiment",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False
        },
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "provider": {
            "organization": "CrewAI"
        }
    }

def run_server():
    """Run the sentiment analysis agent server."""
    try:
        host = "0.0.0.0"
        port = 10000  # Default port for the sentiment analysis agent
        
        # Allow port override via environment variable
        if "SENTIMENT_AGENT_PORT" in os.environ:
            port = int(os.environ["SENTIMENT_AGENT_PORT"])
            
        logger.info(f"Starting sentiment analysis agent server on {host}:{port}")
        
        # Run the server
        uvicorn.run(
            app, 
            host=host, 
            port=port,
            log_level="info"
        )
    except Exception as e:
        logger.exception(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_server()
