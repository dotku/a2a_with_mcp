"""
Agent server module for the sentiment analysis agent.
This module defines the FastAPI app that serves the sentiment analysis agent.
"""

import logging
import os
from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import JSONResponse
import uuid

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

# Delay importing the agent until after PYTHONPATH has been properly set
# This avoids the circular import problem
from sentiment_analysis_agent.crewai.agent import SentimentAnalysisAgent

# Initialize the sentiment analysis agent
sentiment_agent = SentimentAnalysisAgent()

# Define the agent information JSON
agent_info = {
    "name": "Bitcoin Sentiment Analyst",
    "description": "Analyzes Reddit data to determine Bitcoin sentiment",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False
    },
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "provider": {
        "organization": "CrewAI",
        "url": "https://crewai.io"
    },
    "api": {
        "type": "openapi",
        "url": "/openapi.json"
    },
    "url": "http://localhost:10000",
    "version": "1.0.0",
    "skills": [
        {
            "id": "bitcoin-sentiment",
            "name": "Bitcoin Sentiment Analysis",
            "description": "Analyze sentiment from Bitcoin discussions on Reddit",
            "tags": ["bitcoin", "sentiment", "reddit", "analysis"],
            "examples": [
                "What is the current sentiment about Bitcoin on Reddit?",
                "Are Bitcoin discussions positive or negative today?"
            ]
        },
        {
            "id": "reddit-data",
            "name": "Reddit Data Processing",
            "description": "Retrieve and analyze data from Reddit for financial insights",
            "tags": ["reddit", "data", "social media", "crypto"],
            "examples": [
                "What are people saying about Bitcoin on Reddit?",
                "What topics are trending in Bitcoin discussions?"
            ]
        }
    ],
    "documentationUrl": "https://github.com/yourusername/sentiment-analysis-agent"
}

@app.get("/.well-known/agent.json")
async def well_known_agent():
    """Well-known endpoint for agent information."""
    logger.info("Well-known agent info endpoint called")
    return JSONResponse(content=agent_info)

# Add a root endpoint for POST requests - this is what the UI is trying to call
@app.post("/")
async def root_invoke(request: Request):
    """Root endpoint for invoking the agent - handles UI requests."""
    try:
        logger.info("Root endpoint called for agent invocation")
        response = await invoke_agent(request)
        logger.info(f"Successfully processed request and returning response")
        return response
    except Exception as e:
        logger.exception(f"Error in root endpoint: {e}")
        # Get session_id from request if possible, or use default
        session_id = "default_session"
        try:
            data = await request.json()
            session_id = data.get("session_id", session_id)
        except:
            pass
            
        # Create IDs for error response
        task_id = f"task_{session_id}_{uuid.uuid4()}"
        message_id = str(uuid.uuid4())
        
        # Return error in the format that matches what ADKHostManager expects
        return {
            "result": {
                "id": task_id,
                "sessionId": session_id,
                "status": {
                    "state": "failed",
                    "message": {
                        "role": "agent",
                        "parts": [
                            {
                                "type": "text",
                                "text": f"Internal server error: {str(e)}"
                            }
                        ],
                        "metadata": {
                            "message_id": message_id,
                            "conversation_id": session_id,
                            "error": str(e)
                        }
                    }
                },
                "history": [],
                "artifacts": [],
                "metadata": {"conversation_id": session_id}
            }
        }

@router.post("/invoke")
async def invoke_agent(request: Request):
    """Invoke the sentiment analysis agent with the given query."""
    try:
        # Parse the request body
        data = await request.json()
        query = data.get("query", "")
        session_id = data.get("session_id", "default_session")
        
        logger.info(f"Received request to invoke agent with query: {query}, session_id: {session_id}")
        
        # Create a unique task ID
        task_id = f"task_{session_id}_{uuid.uuid4()}"
        
        # Call the synchronous agent
        response = sentiment_agent.invoke(query, session_id)
        
        # Extract the raw string response from CrewOutput
        if hasattr(response, 'raw'):
            response_text = response.raw
        else:
            response_text = str(response)
            
        logger.info(f"Agent response (first 100 chars): {response_text[:100] if response_text else 'None'}")
        
        # Create a message with proper ID for the response
        message_id = str(uuid.uuid4())
        
        # Format the response to EXACTLY match the Task object structure that ADKHostManager expects
        formatted_response = {
            "result": {
                "id": task_id,
                "sessionId": session_id,
                "status": {
                    "state": "completed",
                    "message": {
                        "role": "agent",
                        "parts": [
                            {
                                "type": "text",
                                "text": response_text
                            }
                        ],
                        "metadata": {
                            "message_id": message_id,
                            "conversation_id": session_id
                        }
                    }
                },
                "history": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": query
                            }
                        ],
                        "metadata": {
                            "conversation_id": session_id
                        }
                    }
                ],
                "artifacts": [],
                "metadata": {"conversation_id": session_id}
            }
        }
        
        return formatted_response
    except Exception as e:
        logger.exception(f"Error invoking agent: {e}")
        error_message = f"""# Bitcoin Sentiment Analysis Error

An error occurred while analyzing Bitcoin sentiment: {str(e)}

Please try again later or contact support if this issue persists."""
        
        # Create IDs for error response
        task_id = f"task_{session_id}_{uuid.uuid4()}"
        message_id = str(uuid.uuid4())
        
        # Return error in the format that matches what ADKHostManager expects
        return {
            "result": {
                "id": task_id,
                "sessionId": session_id,
                "status": {
                    "state": "failed",
                    "message": {
                        "role": "agent",
                        "parts": [
                            {
                                "type": "text",
                                "text": error_message
                            }
                        ],
                        "metadata": {
                            "message_id": message_id,
                            "conversation_id": session_id,
                            "error": str(e)
                        }
                    }
                },
                "history": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": query
                            }
                        ],
                        "metadata": {
                            "conversation_id": session_id
                        }
                    }
                ],
                "artifacts": [],
                "metadata": {"conversation_id": session_id}
            }
        }

@router.get("/info")
async def get_agent_info():
    """Return information about this agent for the UI to display."""
    return agent_info

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "OK"} 