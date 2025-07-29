#!/usr/bin/env python3
"""
Server for the sentiment analysis agent.
Implements the A2A protocol to allow communication with other agents.

Usage:
  python server.py

Environment Variables:
  SENTIMENT_AGENT_PORT - Port for the agent server (default: 10000)
  GOOGLE_API_KEY - API key for Google services (required)
"""

import os
import sys
import logging
import json
import traceback
import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Set up absolute imports for the project
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import A2A types from common
from common.types import (
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
    A2ARequest,
    SendTaskRequest,
    SendTaskResponse,
    GetTaskRequest,
    GetTaskResponse,
    CancelTaskRequest,
    CancelTaskResponse,
    SetTaskPushNotificationRequest,
    SetTaskPushNotificationResponse,
    GetTaskPushNotificationRequest,
    GetTaskPushNotificationResponse,
    JSONParseError,
    InvalidRequestError,
    MethodNotFoundError,
    InvalidParamsError,
    InternalError,
    TaskNotFoundError,
    UnsupportedOperationError,
    TaskNotCancelableError,
    AgentCard,
    AgentProvider,
    AgentCapabilities,
    AgentAuthentication,
    AgentSkill,
)

# Import the task manager
from .task_manager import SentimentTaskManager

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

# Create FastAPI app
app = FastAPI(title="Sentiment Analysis Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the TaskManager
task_manager = SentimentTaskManager()

# Define the AgentCard for this service
agent_card = AgentCard(
    name="Bitcoin Sentiment Analyst",
    description="Analyzes Reddit data to determine Bitcoin sentiment",
    url="http://0.0.0.0:10000",
    provider=AgentProvider(
        organization="CrewAI",
        url="https://crewai.example.com"
    ),
    version="1.0.0",
    documentationUrl="https://sentiment-agent.example.com/docs",
    capabilities=AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=True
    ),
    authentication=AgentAuthentication(
        schemes=["none"]
    ),
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    skills=[
        AgentSkill(
            id="sentiment-analysis",
            name="Bitcoin Sentiment Analysis",
            description="Analyze sentiment about Bitcoin from Reddit data",
            tags=["crypto", "sentiment", "bitcoin", "reddit"],
            examples=[
                "What's the current sentiment about Bitcoin on Reddit?",
                "Is the market bullish or bearish on Bitcoin right now?"
            ]
        )
    ]
)

@app.get("/.well-known/agent.json")
async def get_agent_json():
    """
    Return the AgentCard for this service in the agent.json format.
    This is the A2A specification filename.
    """
    logger.info("Serving agent.json")
    return agent_card.model_dump(exclude_none=True)

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    logger.info("Health check request received")
    return {"status": "ok"}

async def handle_json_rpc_request(request: Request) -> JSONRPCResponse:
    """
    Handle a JSON-RPC request.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        JSONRPCResponse
    """
    try:
        # Parse the request body
        body = await request.json()
        logger.info(f"Received JSON-RPC request: {body}")
        
        # Validate the JSON-RPC request
        try:
            rpc_request = A2ARequest.validate_python(body)
            logger.info(f"Validated as {type(rpc_request).__name__}")
            
        except Exception as e:
            logger.error(f"Invalid JSON-RPC request: {e}")
            return JSONRPCResponse(
                id=body.get("id"),
                error=InvalidRequestError(data=str(e))
            )
        
        # Process the request based on the method
        if isinstance(rpc_request, SendTaskRequest):
            # Handle tasks/send method
            logger.info(f"Processing tasks/send for task ID: {rpc_request.params.id}")
            try:
                task = await task_manager.send_task(rpc_request.params)
                logger.info(f"Task created: {task.id}, status: {task.status.state}")
                response = SendTaskResponse(id=rpc_request.id, result=task)
                logger.debug(f"Sending response: {response}")
                return response
            except Exception as e:
                logger.error(f"Error processing send task request: {e}")
                logger.error(traceback.format_exc())
                return JSONRPCResponse(
                    id=rpc_request.id,
                    error=InternalError(data=str(e))
                )
                
        elif isinstance(rpc_request, GetTaskRequest):
            # Handle tasks/get method
            logger.info(f"Processing tasks/get for task ID: {rpc_request.params.id}")
            task = await task_manager.get_task(rpc_request.params)
            if task is None:
                logger.warning(f"Task not found: {rpc_request.params.id}")
                return JSONRPCResponse(
                    id=rpc_request.id,
                    error=TaskNotFoundError()
                )
            logger.info(f"Returning task: {task.id}, status: {task.status.state}")
            return GetTaskResponse(id=rpc_request.id, result=task)
            
        elif isinstance(rpc_request, CancelTaskRequest):
            # Handle tasks/cancel method
            logger.info(f"Processing tasks/cancel for task ID: {rpc_request.params.id}")
            try:
                task = await task_manager.cancel_task(rpc_request.params)
                if task is None:
                    logger.warning(f"Task not found for cancellation: {rpc_request.params.id}")
                    return JSONRPCResponse(
                        id=rpc_request.id,
                        error=TaskNotFoundError()
                    )
                logger.info(f"Canceled task: {task.id}, status: {task.status.state}")
                return CancelTaskResponse(id=rpc_request.id, result=task)
            except TaskNotCancelableError:
                logger.warning(f"Task not cancelable: {rpc_request.params.id}")
                return JSONRPCResponse(
                    id=rpc_request.id,
                    error=TaskNotCancelableError()
                )
            
        elif isinstance(rpc_request, SetTaskPushNotificationRequest):
            # Handle tasks/pushNotification/set method
            logger.info(f"Processing pushNotification/set for task ID: {rpc_request.params.id}")
            try:
                result = await task_manager.set_push_notification(rpc_request.params)
                logger.info(f"Push notification set for task: {result.id}")
                return SetTaskPushNotificationResponse(id=rpc_request.id, result=result)
            except UnsupportedOperationError:
                return JSONRPCResponse(
                    id=rpc_request.id,
                    error=UnsupportedOperationError("Push notifications are not supported")
                )
            
        elif isinstance(rpc_request, GetTaskPushNotificationRequest):
            # Handle tasks/pushNotification/get method
            logger.info(f"Processing pushNotification/get for task ID: {rpc_request.params.id}")
            try:
                result = await task_manager.get_push_notification(rpc_request.params)
                if result is None:
                    logger.warning(f"Push notification not found for task: {rpc_request.params.id}")
                    return JSONRPCResponse(
                        id=rpc_request.id,
                        error=TaskNotFoundError()
                    )
                logger.info(f"Returning push notification for task: {result.id}")
                return GetTaskPushNotificationResponse(id=rpc_request.id, result=result)
            except UnsupportedOperationError:
                return JSONRPCResponse(
                    id=rpc_request.id,
                    error=UnsupportedOperationError("Push notifications are not supported")
                )
            
        else:
            # Unsupported method
            logger.warning(f"Unsupported method: {rpc_request.method}")
            return JSONRPCResponse(
                id=rpc_request.id,
                error=MethodNotFoundError()
            )
            
    except json.JSONDecodeError:
        # Invalid JSON
        logger.error("Invalid JSON in request")
        return JSONRPCResponse(
            id=None,
            error=JSONParseError()
        )
        
    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return JSONRPCResponse(
            id=None,
            error=InternalError(data=str(e))
        )

@app.post("/")
async def json_rpc_endpoint(request: Request) -> JSONRPCResponse:
    """
    Main JSON-RPC endpoint for A2A requests.
    """
    logger.info(f"Incoming request from {request.client.host}")
    response = await handle_json_rpc_request(request)
    logger.info(f"Sending response for request ID: {response.id}")
    logger.debug(f"Response content: {response.model_dump_json()}")
    return response

if __name__ == "__main__":
    # Check if GOOGLE_API_KEY is set
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable must be set.")
        sys.exit(1)
    
    # Run the server
    host = "0.0.0.0"
    port = int(os.environ.get("SENTIMENT_AGENT_PORT", "10000"))
    
    print(f"Starting sentiment analysis agent server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info") 