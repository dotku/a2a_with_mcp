"""
FastAPI server for the Financial Agent.
"""
import logging
import os
import traceback
from typing import Dict, Any, Callable, List, Optional, Union
from uuid import uuid4
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from financial_agent_langgraph.common.types import (
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
    TaskResubscriptionRequest,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
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

from .task_manager import TaskManager

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to add a file handler for persistent logging
try:
    file_handler = logging.FileHandler('financial_agent_server.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info("File logging initialized for server")
except Exception as e:
    logger.error(f"Failed to set up file logging for server: {e}")

# Define the base URL from environment variables
BASE_URL = os.getenv("FIN_AGENT_BASE_URL", "http://localhost:8001/")
logger.info(f"Using base URL: {BASE_URL}")

# Create FastAPI app
app = FastAPI(title="Financial Analysis Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the TaskManager
task_manager = TaskManager()

# Define the AgentCard for this service
agent_card = AgentCard(
    name="Financial Analysis Agent",
    description="Performs financial analysis including valuation metrics, trend analysis, and financial insights.",
    url=BASE_URL,
    provider=AgentProvider(
        organization="Financial Analysts Inc.",
        url="https://financial-analysts.example.com"
    ),
    version="1.0.0",
    documentationUrl="https://financial-agent.example.com/docs",
    capabilities=AgentCapabilities(
        streaming=True,
        pushNotifications=True,
        stateTransitionHistory=True
    ),
    authentication=AgentAuthentication(
        schemes=["none"]
    ),
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    skills=[
        AgentSkill(
            id="financial-metrics",
            name="Financial Metrics Analysis",
            description="Analyze key financial metrics for companies",
            tags=["finance", "metrics", "valuation"],
            examples=[
                "What is the P/E ratio for Apple?",
                "Calculate the intrinsic value of Microsoft shares"
            ]
        ),
        AgentSkill(
            id="trend-analysis",
            name="Financial Trend Analysis",
            description="Analyze financial trends over time",
            tags=["finance", "trends", "time-series"],
            examples=[
                "How has Apple's profit margin changed over the last 3 years?",
                "Show me the revenue growth trend for Microsoft"
            ]
        ),
        AgentSkill(
            id="investment-recommendations",
            name="Investment Recommendations",
            description="Provide investment recommendations based on financial analysis",
            tags=["finance", "investment", "recommendations"],
            examples=[
                "Should I invest in Apple based on their financial metrics?",
                "Give me an investment rating for Microsoft"
            ]
        ),
        AgentSkill(
            id="market-overview",
            name="Market Overview",
            description="Provide an overview of market indices and trends",
            tags=["finance", "market", "indices"],
            examples=[
                "What's the current state of the S&P 500?",
                "How is the NASDAQ performing today?"
            ]
        )
    ]
)

@app.get("/.well-known/ai-plugin.json")
async def get_agent_card():
    """
    Return the AgentCard for this service in the ai-plugin.json format.
    """
    logger.info("Serving ai-plugin.json")
    return agent_card.model_dump(exclude_none=True)

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
            
        except ValidationError as e:
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
            task = await task_manager.cancel_task(rpc_request.params)
            if task is None:
                logger.warning(f"Task not found for cancellation: {rpc_request.params.id}")
                return JSONRPCResponse(
                    id=rpc_request.id,
                    error=TaskNotFoundError()
                )
            logger.info(f"Canceled task: {task.id}, status: {task.status.state}")
            return CancelTaskResponse(id=rpc_request.id, result=task)
            
        elif isinstance(rpc_request, SetTaskPushNotificationRequest):
            # Handle tasks/pushNotification/set method
            logger.info(f"Processing pushNotification/set for task ID: {rpc_request.params.id}")
            result = await task_manager.set_push_notification(rpc_request.params)
            logger.info(f"Push notification set for task: {result.id}")
            return SetTaskPushNotificationResponse(id=rpc_request.id, result=result)
            
        elif isinstance(rpc_request, GetTaskPushNotificationRequest):
            # Handle tasks/pushNotification/get method
            logger.info(f"Processing pushNotification/get for task ID: {rpc_request.params.id}")
            result = await task_manager.get_push_notification(rpc_request.params)
            if result is None:
                logger.warning(f"Push notification not found for task: {rpc_request.params.id}")
                return JSONRPCResponse(
                    id=rpc_request.id,
                    error=TaskNotFoundError()
                )
            logger.info(f"Returning push notification for task: {result.id}")
            return GetTaskPushNotificationResponse(id=rpc_request.id, result=result)
            
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming responses.
    """
    await websocket.accept()
    logger.info(f"WebSocket connection accepted from {websocket.client.host}")
    
    try:
        # Get the initial message
        data = await websocket.receive_text()
        logger.info("Received WebSocket message")
        logger.debug(f"WebSocket message content: {data}")
        
        try:
            # Parse the request
            body = json.loads(data)
            logger.info(f"Parsed WebSocket message as JSON: {body}")
            
            try:
                # Validate the JSON-RPC request
                rpc_request = A2ARequest.validate_python(body)
                logger.info(f"Validated WebSocket message as {type(rpc_request).__name__}")
                
                # Check if this is a streaming request
                if isinstance(rpc_request, SendTaskStreamingRequest):
                    logger.info(f"Processing streaming request for task ID: {rpc_request.params.id}")
                    
                    # Function to send updates to the client
                    async def send_update(update):
                        logger.info(f"Sending update for task: {update.id}, " + 
                                   (f"status: {update.status.state}" if hasattr(update, 'status') else 
                                    f"artifact: {update.artifact.name}" if hasattr(update, 'artifact') else "unknown type"))
                        
                        response = SendTaskStreamingResponse(
                            id=rpc_request.id,
                            result=update
                        )
                        logger.debug(f"Sending WebSocket response: {response.model_dump_json()}")
                        await websocket.send_text(response.model_dump_json())
                        logger.info("WebSocket response sent")
                    
                    # Process the request
                    try:
                        # Send the task
                        logger.info(f"Creating task for streaming request: {rpc_request.params.id}")
                        task = await task_manager.send_task(rpc_request.params)
                        logger.info(f"Task created for streaming: {task.id}, status: {task.status.state}")
                        
                        # Subscribe to updates
                        logger.info(f"Subscribing to updates for task: {task.id}")
                        await task_manager.resubscribe_task(
                            params={"id": task.id},
                            callback=send_update
                        )
                        logger.info(f"Subscribed to updates for task: {task.id}")
                        
                    except Exception as e:
                        logger.error(f"Error processing streaming request: {e}")
                        logger.error(traceback.format_exc())
                        error_response = JSONRPCResponse(
                            id=rpc_request.id,
                            error=InternalError(data=str(e))
                        )
                        logger.debug(f"Sending error response: {error_response.model_dump_json()}")
                        await websocket.send_text(error_response.model_dump_json())
                
                elif isinstance(rpc_request, TaskResubscriptionRequest):
                    logger.info(f"Processing resubscription request for task: {rpc_request.params.id}")
                    
                    # Function to send updates to the client
                    async def send_update(update):
                        logger.info(f"Sending update for resubscribed task: {update.id}")
                        response = SendTaskStreamingResponse(
                            id=rpc_request.id,
                            result=update
                        )
                        logger.debug(f"Sending WebSocket resubscription response: {response.model_dump_json()}")
                        await websocket.send_text(response.model_dump_json())
                    
                    # Resubscribe to task updates
                    logger.info(f"Resubscribing to task: {rpc_request.params.id}")
                    await task_manager.resubscribe_task(
                        params=rpc_request.params,
                        callback=send_update
                    )
                    logger.info(f"Resubscribed to task: {rpc_request.params.id}")
                    
                else:
                    # Unsupported method for WebSocket
                    logger.warning(f"Unsupported WebSocket method: {rpc_request.method}")
                    error_response = JSONRPCResponse(
                        id=body.get("id"),
                        error=UnsupportedOperationError()
                    )
                    logger.debug(f"Sending unsupported method error: {error_response.model_dump_json()}")
                    await websocket.send_text(error_response.model_dump_json())
                    
            except ValidationError as e:
                # Invalid request
                logger.error(f"Invalid WebSocket request: {e}")
                error_response = JSONRPCResponse(
                    id=body.get("id"),
                    error=InvalidRequestError(data=str(e))
                )
                logger.debug(f"Sending validation error: {error_response.model_dump_json()}")
                await websocket.send_text(error_response.model_dump_json())
                
        except json.JSONDecodeError:
            # Invalid JSON
            logger.error("Invalid JSON in WebSocket request")
            error_response = JSONRPCResponse(
                id=None,
                error=JSONParseError()
            )
            logger.debug(f"Sending JSON parse error: {error_response.model_dump_json()}")
            await websocket.send_text(error_response.model_dump_json())
            
    except WebSocketDisconnect:
        logger.info("Client disconnected from WebSocket")
    
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {e}")
        logger.error(traceback.format_exc())
        try:
            error_response = JSONRPCResponse(
                id=None,
                error=InternalError(data=str(e))
            )
            logger.debug(f"Sending unexpected error: {error_response.model_dump_json()}")
            await websocket.send_text(error_response.model_dump_json())
        except Exception as send_error:
            logger.error(f"Error sending WebSocket error response: {send_error}")
            logger.error(traceback.format_exc()) 