import asyncio
import json
import logging
import traceback
from typing import AsyncIterable, Dict, Any, Union

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

# Import the agent implementation
from agent import process_request

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
    
    async def _run_streaming_agent(self, request: SendTaskStreamingRequest):
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)
        
        try:
            # First update - processing started
            task_status = TaskStatus(
                state=TaskState.WORKING,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="Processing orchestration request...")]
                )
            )
            latest_task = await self.update_store(task_send_params.id, task_status, None)
            await self.send_task_notification(latest_task)
            
            # Create task update event
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
            
            task_status = TaskStatus(state=TaskState.COMPLETED)
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
                )
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
        
        # Update task status to working
        task = await self.update_store(
            request.params.id, 
            TaskStatus(state=TaskState.WORKING), 
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