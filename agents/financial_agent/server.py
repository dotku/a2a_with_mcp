"""
FastAPI server for the Financial Agent.
"""
import logging
import os
import traceback
from typing import Dict, Any, Callable, List, Optional, Union
from uuid import uuid4
import json
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

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
    TaskIdParams,
    TaskSendParams,
    Message,
    TextPart,
    Task,
    TaskState,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent
)

# A2A types are not available in this environment, so we'll use fallback implementations

from .task_manager import TaskManager

# Define message-related classes that aren't in common.types
class UITextPart(BaseModel):
    kind: str = "text"
    text: str

class UIMessage(BaseModel):
    kind: str = "message"
    messageId: str | None = None
    parts: list[UITextPart]
    role: str

class MessageSendParams(BaseModel):
    id: str | None = None
    message: UIMessage | Message
    configuration: dict | None = None

class SendMessageRequest(JSONRPCRequest):
    method: str = "message/send"
    params: MessageSendParams

class SendStreamingMessageRequest(JSONRPCRequest):
    method: str = "message/stream"
    params: MessageSendParams

class SendMessageResponse(JSONRPCResponse):
    result: Message

# Message streaming response types are imported from a2a.types above

def create_streaming_message_response(request_id: str | int | None, result: Task | Message | TaskStatusUpdateEvent | TaskArtifactUpdateEvent):
    """Create a properly formatted A2A streaming message response
    
    The A2A client expects: { id, jsonrpc: "2.0", result: Task|Message|TaskStatusUpdateEvent|TaskArtifactUpdateEvent }
    """
    # Ensure result has all required fields for A2A compatibility
    result_dict = result.model_dump() if hasattr(result, 'model_dump') else result
    
    # Add missing required fields based on result type
    if isinstance(result_dict, dict):
        # Add contextId if missing (required for Task, TaskStatusUpdateEvent, TaskArtifactUpdateEvent)
        if 'contextId' not in result_dict and 'context_id' not in result_dict:
            result_dict['contextId'] = str(uuid4())
        
        # For Task objects, ensure contextId is present
        if 'status' in result_dict and 'id' in result_dict:
            if 'contextId' not in result_dict:
                result_dict['contextId'] = str(uuid4())
        
        # For Message objects, ensure messageId is present  
        if 'role' in result_dict and 'parts' in result_dict:
            if 'messageId' not in result_dict and 'message_id' not in result_dict:
                result_dict['messageId'] = str(uuid4())
        
        # For TaskStatusUpdateEvent, ensure required fields
        if 'status' in result_dict and 'final' not in result_dict:
            # This might be a TaskStatusUpdateEvent
            result_dict['final'] = result_dict.get('status', {}).get('state') in ['completed', 'failed', 'canceled']
            if 'taskId' not in result_dict and 'task_id' not in result_dict:
                result_dict['taskId'] = result_dict.get('id', str(uuid4()))
    
    return {
        "id": request_id,
        "jsonrpc": "2.0",
        "result": result_dict
    }

def create_streaming_message_error(request_id: str | int | None, error: JSONRPCError):
    """Create a properly formatted A2A streaming message error response
    
    The A2A client expects: { id, jsonrpc: "2.0", error: JSONRPCError }
    """
    return {
        "id": request_id,
        "jsonrpc": "2.0", 
        "error": error.model_dump() if hasattr(error, 'model_dump') else error
    }

def format_response_as_json(response):
    """Format response as JSON string, handling both Pydantic models and dicts"""
    if isinstance(response, dict):
        return json.dumps(response)
    elif hasattr(response, 'model_dump_json'):
        return response.model_dump_json()
    else:
        return json.dumps(response.model_dump() if hasattr(response, 'model_dump') else response)

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                   format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)

# Try to add a file handler for persistent logging
try:
    file_handler = logging.FileHandler('financial_agent_server.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'))
    logger.addHandler(file_handler)
    logger.info("File logging initialized for server")
except Exception as e:
    logger.error(f"Failed to set up file logging for server: {e}")

# Define the base URL from environment variables
BASE_URL = os.getenv("FIN_AGENT_BASE_URL", "http://localhost:8001")
logger.info(f"Using base URL: {BASE_URL}")

# Create FastAPI app with lifespan events
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Pre-initialize MCP before server is ready
    logger.info("Phase 1: Pre-initializing MCP client before server startup...")
    from agents.financial_agent.agent import ensure_mcp_initialized
    try:
        await ensure_mcp_initialized()
        logger.info("Phase 1 Complete: MCP client successfully pre-initialized")
    except Exception as e:
        logger.error(f"Phase 1 Failed: MCP pre-initialization error: {e}")
        logger.error(traceback.format_exc())
    
    logger.info("Phase 2: FastAPI server ready to accept requests")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Financial Agent server...")
    from agents.financial_agent.agent import cleanup_mcp
    try:
        await cleanup_mcp()
        logger.info("MCP client cleanup completed")
    except Exception as e:
        logger.error(f"Error during MCP cleanup: {e}")

app = FastAPI(title="Financial Analysis Agent", lifespan=lifespan)

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
            # First try to validate as a message request
            method = body.get("method", "")
            if method == "message/send":
                rpc_request = SendMessageRequest.model_validate(body)
                logger.info(f"Validated as SendMessageRequest")
            elif method == "message/stream":
                rpc_request = SendStreamingMessageRequest.model_validate(body)
                logger.info(f"Validated as SendStreamingMessageRequest")
            else:
                # Fall back to task requests
                rpc_request = A2ARequest.validate_python(body)
                logger.info(f"Validated as {type(rpc_request).__name__}")
            
        except ValidationError as e:
            logger.error(f"Invalid JSON-RPC request: {e}")
            return JSONRPCResponse(
                id=body.get("id"),
                error=InvalidRequestError(data=str(e))
            )
        
        # Handle message requests by converting them to task requests
        if isinstance(rpc_request, (SendMessageRequest, SendStreamingMessageRequest)):
            logger.info(f"Processing message request, converting to task")
            
            # Extract text from message - handle both UI and A2A message formats
            message_text = ""
            message = rpc_request.params.message
            
            # Check message type and extract text
            if isinstance(message, UIMessage):
                # Handle UI message format
                for part in message.parts:
                    if part.kind == 'text':
                        message_text = part.text
                        break
            elif isinstance(message, Message):
                # Handle A2A Message format
                for part in message.parts:
                    if hasattr(part, 'text'):
                        message_text = part.text
                        break
            else:
                logger.warning(f"Unknown message type: {type(message)}")
            
            # Create a task from the message
            task_id = rpc_request.params.id or str(uuid4())
            
            # Convert the UI message to an A2A Message for the task
            task_message = Message(
                role="user",
                parts=[TextPart(text=message_text)],
                messageId=str(uuid4()),
                contextId=str(uuid4())
            )
            
            task_params = TaskSendParams(
                id=task_id,
                message=task_message,
                acceptedOutputModes=rpc_request.params.configuration.get('acceptedOutputModes') if rpc_request.params.configuration else None
            )
            
            # Convert to SendTaskStreamingRequest for streaming
            if isinstance(rpc_request, SendStreamingMessageRequest):
                logger.info("Converting message/stream to task streaming request")
                
                async def generate_message_sse_stream():
                    """Generate Server-Sent Events stream for message requests"""
                    try:
                        # Create a queue to receive task updates
                        update_queue = asyncio.Queue()
                        task_completed = False
                        
                        # Define callback for task updates
                        async def task_update_callback(update):
                            await update_queue.put(update)
                        
                        # Subscribe BEFORE creating the task to avoid race conditions
                        task_id = task_params.id
                        await task_manager.resubscribe_task(
                            TaskIdParams(id=task_id),
                            task_update_callback
                        )
                        logger.info(f"Pre-subscribed to updates for task {task_id}")
                        
                        # Now create and process task
                        task = await task_manager.send_task(task_params)
                        logger.info(f"Task created from message: {task.id}, status: {task.status.state}")
                        
                        # Debug: Check if subscription worked
                        subscribers_info = await task_manager.get_all_task_subscribers_details()
                        logger.info(f"All subscribers after subscription: {subscribers_info}")
                        logger.info(f"Subscribers for task {task.id}: {subscribers_info.get(task.id, 'None')}")
                        
                        # Send initial response as a task
                        initial_response = create_streaming_message_response(
                            request_id=rpc_request.id,
                            result=task
                        )
                        
                        # Format as SSE
                        sse_data = f"data: {format_response_as_json(initial_response)}\n\n"
                        yield sse_data
                        
                        # Wait for task updates
                        max_wait_time = 300  # 5 minutes - financial analysis can take time
                        start_time = asyncio.get_event_loop().time()
                        last_keepalive = start_time
                        
                        while not task_completed and (asyncio.get_event_loop().time() - start_time) < max_wait_time:
                            try:
                                # Wait for update with timeout
                                update = await asyncio.wait_for(update_queue.get(), timeout=1.0)
                                logger.info(f"Received update for task {task.id}: {type(update).__name__}")
                                
                                if hasattr(update, 'status'):
                                    # This is a TaskStatusUpdateEvent
                                    task.status = update.status
                                    
                                    # Send task update
                                    update_response = create_streaming_message_response(
                                        request_id=rpc_request.id,
                                        result=task
                                    )
                                    sse_data = f"data: {format_response_as_json(update_response)}\n\n"
                                    yield sse_data
                                    
                                    # Send intermediate status messages for better UX
                                    if update.status.state == TaskState.WORKING:
                                        # Send a processing message
                                        processing_message = Message(
                                            role="assistant", 
                                            parts=[TextPart(text="Processing your financial analysis request... This may take a moment.")],
                                            messageId=str(uuid4()),
                                            contextId=task.contextId if hasattr(task, 'contextId') else str(uuid4())
                                        )
                                        processing_response = create_streaming_message_response(
                                            request_id=rpc_request.id,
                                            result=processing_message
                                        )
                                        sse_data = f"data: {format_response_as_json(processing_response)}\n\n"
                                        yield sse_data
                                    
                                    # If task is complete, send final message
                                    elif update.status.state == TaskState.COMPLETED:
                                        # Extract result from task status message
                                        result_text = "Task completed"
                                        if update.status.message and update.status.message.parts:
                                            for part in update.status.message.parts:
                                                if isinstance(part, dict) and part.get('type') == 'text':
                                                    result_text = part.get('text', result_text)
                                                    break
                                                elif hasattr(part, 'text'):
                                                    result_text = part.text
                                                    break
                                        
                                        final_message = Message(
                                            role="assistant",
                                            parts=[TextPart(text=result_text)],
                                            messageId=str(uuid4()),
                                            contextId=task.contextId if hasattr(task, 'contextId') else str(uuid4())
                                        )
                                        final_response = create_streaming_message_response(
                                            request_id=rpc_request.id,
                                            result=final_message
                                        )
                                        sse_data = f"data: {format_response_as_json(final_response)}\n\n"
                                        yield sse_data
                                        task_completed = True
                                    elif update.status.state in [TaskState.FAILED, TaskState.CANCELED]:
                                        task_completed = True
                                        
                                    # Check if this is the final update
                                    if hasattr(update, 'final') and update.final:
                                        task_completed = True
                                
                            except asyncio.TimeoutError:
                                # No update received in this interval
                                # Send keepalive comment every 10 seconds to prevent connection timeout
                                current_time = asyncio.get_event_loop().time()
                                if current_time - last_keepalive > 10:
                                    # Send a keepalive comment (SSE comments start with :)
                                    yield ": keepalive\n\n"
                                    last_keepalive = current_time
                                    logger.debug(f"Sent keepalive for task {task.id}")
                                continue
                        
                        # Send final done event
                        yield "data: [DONE]\n\n"
                        
                    except Exception as e:
                        logger.error(f"Error in message SSE stream: {e}")
                        error_response = create_streaming_message_error(
                            request_id=rpc_request.id,
                            error=InternalError(data=str(e))
                        )
                        yield f"data: {format_response_as_json(error_response)}\n\n"
                
                # Return SSE streaming response
                return StreamingResponse(
                    generate_message_sse_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    }
                )
            else:
                # Handle non-streaming message request
                logger.info("Converting message/send to task request")
                task = await task_manager.send_task(task_params)
                
                # Wait for task completion
                max_wait_time = 300  # 5 minutes - financial analysis can take time
                check_interval = 1  # second
                elapsed_time = 0
                
                while elapsed_time < max_wait_time:
                    current_task = task_manager.tasks.get(task.id)
                    if current_task and current_task.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
                        task = current_task
                        break
                    await asyncio.sleep(check_interval)
                    elapsed_time += check_interval
                
                # Extract result from task status message
                result_text = "Processing your request..."
                if task.status.state == TaskState.COMPLETED and task.status.message and task.status.message.parts:
                    for part in task.status.message.parts:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            result_text = part.get('text', result_text)
                            break
                        elif hasattr(part, 'text'):
                            result_text = part.text
                            break
                elif task.status.state == TaskState.FAILED:
                    result_text = "Task failed"
                
                response_message = Message(
                    role="assistant",
                    parts=[TextPart(text=result_text)],
                    messageId=str(uuid4()),
                    contextId=str(uuid4())
                )
                
                return SendMessageResponse(id=rpc_request.id, result=response_message)
        
        # Handle SendTaskStreamingRequest with SSE
        elif isinstance(rpc_request, SendTaskStreamingRequest):
            logger.info(f"Processing SendTaskStreamingRequest for task ID: {rpc_request.params.id}")
            
            async def generate_sse_stream():
                """Generate Server-Sent Events stream"""
                try:
                    # Create task
                    task = await task_manager.send_task(rpc_request.params)
                    logger.info(f"Task created: {task.id}, status: {task.status.state}")
                    
                    # Send initial status
                    initial_response = SendTaskStreamingResponse(
                        id=rpc_request.id,
                        result=task
                    )
                    
                    # Format as SSE
                    sse_data = f"data: {initial_response.model_dump_json()}\n\n"
                    yield sse_data
                    
                    # Wait for task completion and send updates
                    max_wait_time = 30  # seconds
                    check_interval = 1  # second
                    elapsed_time = 0
                    
                    while elapsed_time < max_wait_time:
                        current_task = task_manager.tasks.get(task.id)
                        if current_task and current_task.status.state != task.status.state:
                            # Task status changed, send update
                            task = current_task
                            update_response = SendTaskStreamingResponse(
                                id=rpc_request.id,
                                result=task
                            )
                            sse_data = f"data: {update_response.model_dump_json()}\n\n"
                            yield sse_data
                            
                            # If task is complete, break
                            if current_task.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
                                break
                        
                        await asyncio.sleep(check_interval)
                        elapsed_time += check_interval
                    
                    # Send final done event
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}")
                    error_response = JSONRPCResponse(
                        id=rpc_request.id,
                        error=InternalError(data=str(e))
                    )
                    yield f"data: {error_response.model_dump_json()}\n\n"
            
            # Return SSE streaming response
            return StreamingResponse(
                generate_sse_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                }
            )
        
        # Handle other request types (existing code)
        elif isinstance(rpc_request, SendTaskRequest):
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
        elif isinstance(rpc_request, SendTaskStreamingRequest):
            # Handle streaming requests as regular tasks over HTTP
            logger.info("Processing SendTaskStreamingRequest as regular task over HTTP")
            try:
                task = await task_manager.send_task(rpc_request.params)
                logger.info(f"Task created: {task.id}, status: {task.status.state}")
                
                # Wait for task completion instead of subscribing
                max_wait_time = 30  # seconds
                check_interval = 1  # second
                elapsed_time = 0
                
                while elapsed_time < max_wait_time:
                    current_task = task_manager.tasks.get(task.id)
                    if current_task and current_task.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
                        task = current_task
                        break
                    await asyncio.sleep(check_interval)
                    elapsed_time += check_interval
                
                response = SendTaskResponse(id=rpc_request.id, result=task)
                # response = await task_manager.on_send_task_subscribe(request)
                async def send_update(update):
                    logger.info(f"Sending update for task: {update.id}, " + 
                                (f"status: {update.status.state}" if hasattr(update, 'status') else 
                                f"artifact: {update.artifact.name}" if hasattr(update, 'artifact') else "unknown type"))
                SendTaskStreamingResponse(
                            id=rpc_request.params.id,
                            result=send_update
                        )
                
            except Exception as e:
                logger.error(f"Error processing streaming task: {e}")
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
async def json_rpc_endpoint(request: Request):
    """
    Main JSON-RPC endpoint for A2A requests.
    """
    logger.info(f"Incoming request from {request.client.host}")
    # Use the common handler
    return await handle_json_rpc_request(request)

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
                            id=rpc_request.params.id,
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
                            params=TaskIdParams(id=task.id),
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