import asyncio
import json
import logging
import traceback
from typing import AsyncIterable, Dict, Any, Union
from datetime import datetime, timezone
import uuid
import os

from common.server.task_manager import InMemoryTaskManager
from common.types import (
    SendTaskRequest,
    TaskSendParams,
    Message,
    TaskStatus,
    Artifact,
    TextPart,
    TaskState,
    SendTaskResponse,
    InternalError,
    JSONRPCResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    Task,
    TaskIdParams,
    PushNotificationConfig,
    SetTaskPushNotificationRequest,
    SetTaskPushNotificationResponse,
    TaskPushNotificationConfig,
    TaskNotFoundError,
    InvalidParamsError,
)
import common.server.utils as utils

# Use relative import for sibling module
from .agent import process_request

logger = logging.getLogger(__name__)

# Simple implementation of PushNotificationSenderAuth
class PushNotificationSenderAuth:
    """
    Simplified PushNotificationSenderAuth implementation.
    In a real implementation, this would handle JWK authentication.
    """
    def __init__(self):
        self.jwk = {}
    
    def generate_jwk(self):
        """Generate a JSON Web Key (simplified)."""
        self.jwk = {"kid": "test-key-1", "alg": "HS256"}
        logger.info("Generated JWK (simplified for demo)")
    
    async def verify_push_notification_url(self, url: str) -> bool:
        """Always return True for demo purposes."""
        logger.info(f"Verifying push notification URL: {url} (simplified)")
        return True
    
    async def send_push_notification(self, url: str, data: Dict[str, Any]) -> bool:
        """Simulate sending a push notification."""
        logger.info(f"Sending push notification to {url} (simplified)")
        return True
    
    async def handle_jwks_endpoint(self, request):
        """Handle JWKs endpoint for authentication."""
        return {"keys": [self.jwk]}

class OrchestratorTaskManager(InMemoryTaskManager):
    def __init__(self, notification_sender_auth: PushNotificationSenderAuth):
        super().__init__()
        self.notification_sender_auth = notification_sender_auth
        # Store remote agents' push URLs
        self.remote_agent_capabilities = {}
        # Store delegated task mappings (local task ID -> remote task ID)
        self.delegated_tasks = {}
    
    async def _run_streaming_agent(self, request: SendTaskStreamingRequest):
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)
        
        try:
            # First update - SUBMITTED state
            task_status = TaskStatus(
                state=TaskState.SUBMITTED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="Task submitted to orchestrator...")]
                ),
                timestamp=datetime.now(timezone.utc)
            )
            latest_task = await self.update_store(task_send_params.id, task_status, None)
            await self.send_task_notification(latest_task)
            
            # Create task update event for SUBMITTED state
            task_update_event = TaskStatusUpdateEvent(
                id=task_send_params.id,
                status=task_status,
                final=False
            )
            await self.enqueue_events_for_sse(task_send_params.id, task_update_event)
            
            # Then update to WORKING state
            task_status = TaskStatus(
                state=TaskState.WORKING,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="Processing orchestration request...")]
                ),
                timestamp=datetime.now(timezone.utc)
            )
            latest_task = await self.update_store(task_send_params.id, task_status, None)
            await self.send_task_notification(latest_task)
            
            # Create task update event for WORKING state
            task_update_event = TaskStatusUpdateEvent(
                id=task_send_params.id,
                status=task_status,
                final=False
            )
            await self.enqueue_events_for_sse(task_send_params.id, task_update_event)
            
            # Process the request
            input_data = self._extract_input_data(task_send_params)
            result = process_request(input_data)
            
            # Second update - results ready
            artifact = Artifact(
                parts=[TextPart(text=json.dumps(result))],
                index=0,
                append=False
            )
            
            task_status = TaskStatus(state=TaskState.COMPLETED, timestamp=datetime.now(timezone.utc))
            latest_task = await self.update_store(
                task_send_params.id,
                task_status,
                [artifact]
            )
            await self.send_task_notification(latest_task)
            
            # Send artifact update
            task_artifact_update_event = TaskArtifactUpdateEvent(
                id=task_send_params.id,
                artifact=artifact
            )
            await self.enqueue_events_for_sse(task_send_params.id, task_artifact_update_event)
            
            # Send final status update
            task_update_event = TaskStatusUpdateEvent(
                id=task_send_params.id,
                status=task_status,
                final=True
            )
            await self.enqueue_events_for_sse(task_send_params.id, task_update_event)
            
        except Exception as e:
            logger.error(f"An error occurred while streaming the response: {e}")
            task_status = TaskStatus(
                state=TaskState.FAILED,
                message=Message(
                    role="agent", 
                    parts=[TextPart(text=f"Error processing request: {str(e)}")]
                ),
                timestamp=datetime.now(timezone.utc)
            )
            await self.update_store(task_send_params.id, task_status, None)
            
            await self.enqueue_events_for_sse(
                task_send_params.id,
                InternalError(message=f"An error occurred while streaming the response: {e}")
            )
    
    def _validate_request(self, request: Union[SendTaskRequest, SendTaskStreamingRequest]) -> JSONRPCResponse | None:
        task_send_params: TaskSendParams = request.params
        
        # Validate supported content types
        supported_content_types = ["text", "text/plain"]
        if not utils.are_modalities_compatible(
            task_send_params.acceptedOutputModes, supported_content_types
        ):
            logger.warning(
                "Unsupported output mode. Received %s, Support %s",
                task_send_params.acceptedOutputModes,
                supported_content_types
            )
            return utils.new_incompatible_types_error(request.id)
        
        # Validate push notification
        if task_send_params.pushNotification and not task_send_params.pushNotification.url:
            logger.warning("Push notification URL is missing")
            return JSONRPCResponse(
                id=request.id, 
                error=InvalidParamsError(message="Push notification URL is missing")
            )
        
        return None
    
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handles the 'send task' request."""
        validation_error = self._validate_request(request)
        if validation_error:
            return SendTaskResponse(id=request.id, error=validation_error.error)
        
        # Set up push notification if provided
        if request.params.pushNotification:
            if not await self.set_push_notification_info(request.params.id, request.params.pushNotification):
                return SendTaskResponse(
                    id=request.id, 
                    error=InvalidParamsError(message="Push notification URL is invalid")
                )

        # Initialize the task
        await self.upsert_task(request.params)
        
        # First set task status to SUBMITTED
        task = await self.update_store(
            request.params.id, 
            TaskStatus(state=TaskState.SUBMITTED, timestamp=datetime.now(timezone.utc)), 
            None
        )
        await self.send_task_notification(task)
        
        # Then update to WORKING
        task = await self.update_store(
            request.params.id, 
            TaskStatus(state=TaskState.WORKING, timestamp=datetime.now(timezone.utc)), 
            None
        )
        await self.send_task_notification(task)

        try:
            # Extract the query from the request
            input_data = self._extract_input_data(request.params)
            
            # Process the request synchronously
            result = process_request(input_data)
            
            # Create message and artifact
            parts = [TextPart(text=json.dumps(result))]
            
            # Update task with completed status and artifact
            task_status = TaskStatus(state=TaskState.COMPLETED)
            artifact = Artifact(parts=parts)
            
            task = await self.update_store(
                request.params.id,
                task_status,
                [artifact]
            )
            
            # Include history if requested
            task_result = self.append_task_history(task, request.params.historyLength)
            
            # Send push notification if configured
            await self.send_task_notification(task)
            
            return SendTaskResponse(id=request.id, result=task_result)
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            logger.error(traceback.format_exc())
            
            # Update task with error status
            error_message = Message(
                role="agent",
                parts=[TextPart(text=f"Error: {str(e)}")]
            )
            error_status = TaskStatus(state=TaskState.FAILED, message=error_message)
            
            await self.update_store(request.params.id, error_status, None)
            
            return SendTaskResponse(
                id=request.id,
                error=InternalError(message=f"Error processing task: {str(e)}")
            )
    
    async def on_send_task_subscribe(self, request: SendTaskStreamingRequest) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        """Handles the 'send task subscribe' request for streaming."""
        try:
            error = self._validate_request(request)
            if error:
                return error

            # Initialize the task
            await self.upsert_task(request.params)

            # Set up push notification if provided
            if request.params.pushNotification:
                if not await self.set_push_notification_info(request.params.id, request.params.pushNotification):
                    return JSONRPCResponse(
                        id=request.id, 
                        error=InvalidParamsError(message="Push notification URL is invalid")
                    )

            # Set up SSE consumer
            sse_event_queue = await self.setup_sse_consumer(request.params.id, False)
            
            # Start the streaming process in a background task
            asyncio.create_task(self._run_streaming_agent(request))

            # Return a generator that will stream SSE events
            return self.dequeue_events_for_sse(
                request.id, 
                request.params.id,
                sse_event_queue
            )
            
        except Exception as e:
            logger.error(f"Error in SSE stream: {e}")
            print(traceback.format_exc())
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message=f"An error occurred while streaming the response: {str(e)}"
                ),
            )
    
    async def on_resubscribe_to_task(self, request) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        """Handles task resubscription."""
        task_id_params: TaskIdParams = request.params
        try:
            sse_event_queue = await self.setup_sse_consumer(task_id_params.id, True)
            return self.dequeue_events_for_sse(request.id, task_id_params.id, sse_event_queue)
        except Exception as e:
            logger.error(f"Error while reconnecting to SSE stream: {e}")
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message=f"An error occurred while reconnecting to stream: {e}"
                ),
            )
    
    async def set_push_notification_info(self, task_id: str, push_notification_config: PushNotificationConfig):
        """Set up push notification and verify URL ownership."""
        # Verify the ownership of notification URL by issuing a challenge request
        is_verified = await self.notification_sender_auth.verify_push_notification_url(
            push_notification_config.url
        )
        if not is_verified:
            return False
        
        await super().set_push_notification_info(task_id, push_notification_config)
        return True
    
    async def send_task_notification(self, task: Task):
        """Send push notification for task updates."""
        if not await self.has_push_notification_info(task.id):
            logger.info(f"No push notification info found for task {task.id}")
            return
        
        push_info = await self.get_push_notification_info(task.id)

        logger.info(f"Notifying for task {task.id} => {task.status.state}")
        await self.notification_sender_auth.send_push_notification(
            push_info.url,
            data=task.model_dump(exclude_none=True)
        )
    
    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        """Extract user text from message parts."""
        for part in task_send_params.message.parts:
            if hasattr(part, 'type') and part.type == "text":
                return part.text
        return ""
    
    def _extract_input_data(self, task_send_params: TaskSendParams) -> Dict[str, Any]:
        """Extract input data from the task parameters."""
        input_data = {}
        
        # Extract query from message parts
        query = self._get_user_query(task_send_params)
        
        # Try to parse as JSON if it looks like JSON
        if query.strip().startswith('{') and query.strip().endswith('}'):
            try:
                input_data = json.loads(query)
            except json.JSONDecodeError:
                input_data = {"query": query}
        else:
            input_data = {"query": query}
        
        # Add metadata
        if task_send_params.metadata:
            input_data["metadata"] = task_send_params.metadata
        
        return input_data 
    
    async def handle_push_callback(self, task_id: str, update_data: Dict[str, Any]):
        """
        Handle a push callback from a remote agent.
        
        Args:
            task_id: The task ID for the callback
            update_data: The update data from the remote agent
        """
        logger.info(f"Received push callback for task: {task_id}")
        logger.debug(f"Push callback data: {update_data}")
        
        # Find the parent task ID for this delegated task
        parent_task_id = None
        for local_id, remote_id in self.delegated_tasks.items():
            if remote_id == task_id:
                parent_task_id = local_id
                break
        
        if not parent_task_id:
            logger.warning(f"Received push callback for unknown delegated task: {task_id}")
            return
        
        # Update the parent task based on the remote update
        try:
            # Determine if this is a status update or artifact update
            if "status" in update_data:
                # Status update
                remote_status = update_data["status"]
                task_status = TaskStatus(
                    state=remote_status.get("state", TaskState.WORKING),
                    message=remote_status.get("message"),
                    timestamp=datetime.now(timezone.utc)
                )
                
                if task_status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
                    # Final update
                    is_final = True
                else:
                    is_final = False
                
                await self.update_store(parent_task_id, task_status, None)
                
                # Create and enqueue a status update event
                status_event = TaskStatusUpdateEvent(
                    id=parent_task_id,
                    status=task_status,
                    final=is_final
                )
                await self.enqueue_events_for_sse(parent_task_id, status_event)
                
            elif "artifact" in update_data:
                # Artifact update
                remote_artifact = update_data["artifact"]
                artifact = Artifact(
                    name=remote_artifact.get("name", "delegated-result"),
                    parts=remote_artifact.get("parts", []),
                    index=remote_artifact.get("index", 0),
                    append=remote_artifact.get("append", False)
                )
                
                # Get the current task to add the artifact
                task = await self.get_task(TaskIdParams(id=parent_task_id))
                if task:
                    if not task.artifacts:
                        task.artifacts = []
                    task.artifacts.append(artifact)
                    await self.upsert_task(task)
                
                # Create and enqueue an artifact update event
                artifact_event = TaskArtifactUpdateEvent(
                    id=parent_task_id,
                    artifact=artifact
                )
                await self.enqueue_events_for_sse(parent_task_id, artifact_event)
        
        except Exception as e:
            logger.error(f"Error processing push callback for task {task_id}: {e}")
            logger.error(traceback.format_exc())
    
    async def delegate_to_remote_agent(self, agent_url: str, task_data: Dict[str, Any], parent_task_id: str) -> str:
        """
        Delegate a task to a remote agent, preferring push notifications if supported.
        
        Args:
            agent_url: URL of the remote agent
            task_data: Task data to send to the remote agent
            parent_task_id: ID of the parent task
            
        Returns:
            The ID of the delegated task
        """
        import httpx
        import json
        
        # Check if we have cached capability information for this agent
        if agent_url not in self.remote_agent_capabilities:
            # Fetch the agent's capabilities
            try:
                agent_base_url = agent_url.rstrip("/")
                agent_card_url = f"{agent_base_url}/.well-known/agent.json"
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(agent_card_url)
                    if response.status_code == 200:
                        agent_card = response.json()
                        self.remote_agent_capabilities[agent_url] = agent_card.get("capabilities", {})
                        logger.info(f"Cached capabilities for agent {agent_url}: {self.remote_agent_capabilities[agent_url]}")
                    else:
                        logger.warning(f"Failed to fetch capabilities for agent {agent_url}, status: {response.status_code}")
                        self.remote_agent_capabilities[agent_url] = {}
            except Exception as e:
                logger.error(f"Error fetching agent capabilities for {agent_url}: {e}")
                self.remote_agent_capabilities[agent_url] = {}
        
        # Generate a task ID for the delegated task
        delegated_task_id = f"delegated-{uuid.uuid4()}"
        
        # Prepare the JSON-RPC request
        rpc_method = "tasks/send"
        push_notification = None
        
        # Check if the remote agent supports push notifications
        supports_push = self.remote_agent_capabilities.get(agent_url, {}).get("pushNotifications", False)
        
        # If push notifications are supported, set up a push URL
        if supports_push:
            # Calculate our callback URL
            host = os.environ.get("HOST", "localhost")
            port = os.environ.get("PORT", "8000")
            push_url = f"http://{host}:{port}/push/{delegated_task_id}"
            
            # Set up push notification
            push_notification = {
                "url": push_url
            }
            logger.info(f"Using push notifications for delegated task {delegated_task_id} to {agent_url}")
        else:
            logger.info(f"Remote agent {agent_url} does not support push notifications, will poll for updates")
        
        # Prepare the task parameters
        task_params = {
            "id": delegated_task_id,
            "sessionId": parent_task_id,  # Use parent task ID as session ID for tracking
            "message": task_data["message"]
        }
        
        # Add push notification if supported
        if push_notification:
            task_params["pushNotification"] = push_notification
        
        # Prepare the JSON-RPC request
        rpc_request = {
            "jsonrpc": "2.0",
            "method": rpc_method,
            "params": task_params,
            "id": str(uuid.uuid4())
        }
        
        # Send the request to the remote agent
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    agent_url,
                    json=rpc_request,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code >= 200 and response.status_code < 300:
                    # Store the mapping between our task ID and the delegated task ID
                    self.delegated_tasks[parent_task_id] = delegated_task_id
                    logger.info(f"Successfully delegated task {parent_task_id} to {agent_url} as {delegated_task_id}")
                    return delegated_task_id
                else:
                    logger.error(f"Error delegating task to {agent_url}: {response.text}")
                    raise Exception(f"Error delegating task: status {response.status_code}")
        except Exception as e:
            logger.error(f"Error delegating task to {agent_url}: {e}")
            raise 