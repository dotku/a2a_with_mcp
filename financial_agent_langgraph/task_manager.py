"""
Task Manager for Financial Agent
"""
import asyncio
import logging
import json
import traceback
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
from datetime import datetime
from uuid import uuid4

from financial_agent_langgraph.common.types import (
    Task,
    TaskStatus,
    TaskState,
    TaskIdParams,
    TaskSendParams,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    Artifact,
    Message,
    PushNotificationConfig,
    TaskPushNotificationConfig,
)

from .agent import process_financial_task, process_financial_task_async

# Configure more detailed logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Explicitly set DEBUG level for this logger
logger.info("task_manager logger explicitly set to DEBUG level")

class TaskManager:
    """
    Task Manager for the Financial Agent.
    
    This class handles the lifecycle of tasks, including:
    - Creating tasks
    - Updating task status
    - Managing push notifications
    - Processing tasks using the financial agent
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.push_notification_configs: Dict[str, PushNotificationConfig] = {}
        self.task_subscribers: Dict[str, Set[Callable]] = {}
        self.lock = asyncio.Lock()
        logger.info("TaskManager initialized")
    
    async def get_task(self, params: TaskIdParams) -> Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            params: TaskIdParams containing the task ID
            
        Returns:
            The task if found, None otherwise
        """
        logger.info(f"Get task request for ID: {params.id}")
        async with self.lock:
            if params.id not in self.tasks:
                logger.warning(f"Task {params.id} not found")
                return None
            
            task = self.tasks[params.id]
            logger.info(f"Returning task {params.id}, state: {task.status.state}")
            return task
    
    async def send_task(self, params: TaskSendParams) -> Task:
        """
        Create and send a new task for financial analysis.
        
        Args:
            params: TaskSendParams containing task information
            
        Returns:
            The created task
        """
        task_id = params.id
        session_id = params.sessionId
        message = params.message
        push_notification = params.pushNotification
        
        logger.info(f"New task request: ID={task_id}, Session={session_id}")
        logger.debug(f"Message content: {message}")
        
        # Extract message text for logging
        message_text = ""
        for part in message.parts:
            if hasattr(part, "type") and part.type == "text":
                message_text += part.text
            elif isinstance(part, dict) and part.get("type") == "text":
                message_text += part.get("text", "")
        
        logger.info(f"Task message: '{message_text}'")
        
        # Initialize task history with the incoming message
        history = [message]
        
        # Create the task
        task = Task(
            id=task_id,
            sessionId=session_id,
            status=TaskStatus(
                state=TaskState.SUBMITTED,
                timestamp=datetime.now()
            ),
            history=history,
            metadata=params.metadata
        )
        
        logger.info(f"Created task {task_id} with status: {task.status.state}")
        
        # Store the task
        async with self.lock:
            self.tasks[task_id] = task
            
            # Set up push notification if provided
            if push_notification:
                logger.info(f"Setting up push notification for task {task_id}")
                self.push_notification_configs[task_id] = push_notification
        
        # Publish initial status update to subscribers
        logger.info(f"Publishing initial status update for task {task_id}")
        asyncio.create_task(self._publish_status_update(task, False))
        
        # Process the task asynchronously
        logger.info(f"Creating async task to process task {task_id}")
        asyncio.create_task(self._process_task(task_id))
        
        return task
    
    async def cancel_task(self, params: TaskIdParams) -> Optional[Task]:
        """
        Cancel a task by ID.
        
        Args:
            params: TaskIdParams containing the task ID
            
        Returns:
            The updated task if found and canceled, None otherwise
        """
        logger.info(f"Cancel task request for ID: {params.id}")
        async with self.lock:
            if params.id not in self.tasks:
                logger.warning(f"Task {params.id} not found for cancellation")
                return None
            
            task = self.tasks[params.id]
            
            # Only tasks in SUBMITTED or WORKING state can be canceled
            if task.status.state not in [TaskState.SUBMITTED, TaskState.WORKING]:
                logger.info(f"Task {params.id} cannot be canceled - current state: {task.status.state}")
                return task
            
            # Update task status to CANCELED
            task.status = TaskStatus(
                state=TaskState.CANCELED,
                timestamp=datetime.now()
            )
            
            logger.info(f"Task {params.id} has been canceled")
            self.tasks[params.id] = task
            
            # Publish status update to subscribers
            await self._publish_status_update(task, True)
            
            return task
    
    async def set_push_notification(self, params: TaskPushNotificationConfig) -> TaskPushNotificationConfig:
        """
        Set push notification configuration for a task.
        
        Args:
            params: TaskPushNotificationConfig containing task ID and push notification config
            
        Returns:
            The provided TaskPushNotificationConfig
        """
        logger.info(f"Setting push notification for task: {params.id}")
        async with self.lock:
            self.push_notification_configs[params.id] = params.pushNotificationConfig
        
        return params
    
    async def get_push_notification(self, params: TaskIdParams) -> Optional[TaskPushNotificationConfig]:
        """
        Get push notification configuration for a task.
        
        Args:
            params: TaskIdParams containing the task ID
            
        Returns:
            TaskPushNotificationConfig if found, None otherwise
        """
        logger.info(f"Get push notification request for task: {params.id}")
        async with self.lock:
            if params.id not in self.push_notification_configs:
                logger.warning(f"No push notification found for task {params.id}")
                return None
            
            config = self.push_notification_configs[params.id]
            logger.info(f"Returning push notification config for task {params.id}")
            
            return TaskPushNotificationConfig(
                id=params.id,
                pushNotificationConfig=config
            )
    
    async def resubscribe_task(self, params: TaskIdParams, callback: Callable) -> None:
        """
        Resubscribe to task updates.
        
        Args:
            params: TaskIdParams containing the task ID
            callback: Function to call on task updates
        """
        task_id = params.id
        logger.info(f"Resubscribe request for task: {task_id}")
        
        async with self.lock:
            if task_id not in self.task_subscribers:
                self.task_subscribers[task_id] = set()
            
            self.task_subscribers[task_id].add(callback)
            logger.info(f"Added subscriber to task {task_id}, total subscribers: {len(self.task_subscribers[task_id])}")
            
            # If the task exists, immediately publish its current status
            if task_id in self.tasks:
                task = self.tasks[task_id]
                logger.info(f"Task {task_id} exists, publishing current status (state: {task.status.state})")
                
                # Create a status update event
                status_event = TaskStatusUpdateEvent(
                    id=task_id,
                    status=task.status,
                    final=task.status.state in [
                        TaskState.COMPLETED,
                        TaskState.CANCELED,
                        TaskState.FAILED
                    ],
                    metadata=task.metadata
                )
                
                # Call the callback with the status event
                try:
                    logger.info(f"Calling status callback for task {task_id}")
                    callback(status_event)
                except Exception as e:
                    logger.error(f"Error calling status callback for task {task_id}: {e}")
                    logger.error(traceback.format_exc())
                
                # Send any artifacts
                if task.artifacts:
                    logger.info(f"Task {task_id} has {len(task.artifacts)} artifacts to send")
                    for artifact in task.artifacts:
                        artifact_event = TaskArtifactUpdateEvent(
                            id=task_id,
                            artifact=artifact,
                            metadata=task.metadata
                        )
                        
                        try:
                            logger.info(f"Calling artifact callback for task {task_id}, artifact: {artifact.name}")
                            callback(artifact_event)
                        except Exception as e:
                            logger.error(f"Error calling artifact callback for task {task_id}: {e}")
                            logger.error(traceback.format_exc())
    
    async def _process_task(self, task_id: str) -> None:
        """
        Process a task asynchronously.
        
        Args:
            task_id: ID of the task to process
        """
        logger.info(f"Processing task {task_id}")
        
        task_to_publish = None
        async with self.lock: # Lock for initial task state update
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found for processing")
                return
            
            task = self.tasks[task_id]
            
            # Update task status to WORKING
            task.status = TaskStatus(
                state=TaskState.WORKING,
                timestamp=datetime.now()
            )
            self.tasks[task_id] = task
            task_to_publish = task # Prepare task for publishing after lock release
            logger.info(f"Updated task {task_id} status to WORKING within lock")
        
        # Publish status update AFTER releasing the initial lock
        if task_to_publish:
            logger.info(f"Publishing WORKING state for task {task_id} (outside initial lock)")
            await self._publish_status_update(task_to_publish, False)
            logger.info(f"Published WORKING state for task {task_id}, sleeping briefly to allow orchestrator to see update")
            await asyncio.sleep(0.5)
        else: # Should not happen if task was found
            logger.error(f"Task {task_id} was None after initial lock release, cannot publish WORKING state.")
            return

        current_task_for_processing = task_to_publish # Use the consistent task object

        try:
            logger.info(f"Processing task {task_id} with message: '{current_task_for_processing.history[-1].parts[0].text}'")
            # Add log before starting the financial task
            logger.debug(f"About to call process_financial_task_async for task {task_id}")
            try:
                # Add a timeout to the financial task processing (e.g., 60 seconds)
                updated_task = await asyncio.wait_for(
                    process_financial_task_async(current_task_for_processing),
                    timeout=60
                )
                logger.debug(f"process_financial_task completed for task {task_id}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout while processing task {task_id}")
                async with self.lock:
                    if task_id in self.tasks:
                        task = self.tasks[task_id]
                        task.status = TaskStatus(
                            state=TaskState.FAILED,
                            message=Message(
                                role="agent",
                                parts=[{"type": "text", "text": "Task timed out after 60 seconds."}]
                            ),
                            timestamp=datetime.now()
                        )
                        logger.info(f"Updated task {task_id} status to FAILED due to timeout")
                        self.tasks[task_id] = task
                    # Lock is released by async with
                
                # Publish FAILED state (this will re-acquire lock inside _publish_status_update)
                # Need to fetch the task again as it might have been modified by another coroutine if we released the lock earlier
                # However, for timeout, we are the sole modifier of this task's failure state here.
                # For safety, let's re-fetch if possible, or use the locally modified one if re-fetch fails.
                failed_task_to_publish = None
                async with self.lock:
                    if task_id in self.tasks:
                        failed_task_to_publish = self.tasks[task_id] # it should be the one we just modified
                
                if failed_task_to_publish:
                     await self._publish_status_update(failed_task_to_publish, True)
                else: # Fallback to the task object we have if it's gone from self.tasks for some reason
                    # This is less ideal as it might not be the absolute latest from shared state
                    # but it's the one we know is FAILED.
                    async with self.lock: # Re-acquire to get a potentially updated task object
                         task_for_timeout_fail_report = self.tasks.get(task_id)
                    if task_for_timeout_fail_report: # If it still exists
                         task_for_timeout_fail_report.status = TaskStatus(
                              state=TaskState.FAILED,
                              message=Message(role="agent", parts=[{"type": "text", "text": "Task timed out after 60 seconds."}]),
                              timestamp=datetime.now()
                         )
                         await self._publish_status_update(task_for_timeout_fail_report, True)
                    else:
                         logger.error(f"Task {task_id} not found for publishing FAILED timeout status.")
                return
            response_text = updated_task.status.message.parts[0].text if updated_task.status.message and updated_task.status.message.parts else ""
            artifacts = updated_task.artifacts or []
            logger.info(f"Received response for task {task_id}: '{response_text[:100]}...'")
            
            # Prepare artifacts
            artifact_objects = []
            for artifact in artifacts:
                name = getattr(artifact, "name", "financial-data")
                description = getattr(artifact, "description", "Financial analysis data")
                # If artifact.parts is a list of dicts or TextPart, extract text
                content = ""
                if hasattr(artifact, "parts") and artifact.parts:
                    for part in artifact.parts:
                        if hasattr(part, "type") and part.type == "text":
                            content += getattr(part, "text", "")
                        elif isinstance(part, dict) and part.get("type") == "text":
                            content += part.get("text", "")
                artifact_obj = Artifact(
                    name=name,
                    description=description,
                    parts=[{"type": "text", "text": content}]
                )
                logger.info(f"Created artifact for task {task_id}: {name}")
                artifact_objects.append(artifact_obj)
            
            async with self.lock: # Lock for final COMPLETED state update
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    # Add artifacts to task
                    task.artifacts = artifact_objects
                    # Create agent message from response
                    assistant_message = Message(
                        role="agent",
                        parts=[{"type": "text", "text": response_text}]
                    )
                    # Add response to history
                    task.history.append(assistant_message)
                    # Update task status to COMPLETED
                    task.status = TaskStatus(
                        state=TaskState.COMPLETED,
                        message=assistant_message,
                        timestamp=datetime.now()
                    )
                    logger.info(f"Updated task {task_id} status to COMPLETED")
                    self.tasks[task_id] = task
                    completed_task_to_publish = task # Prepare for publishing after lock release
                else:
                    logger.warning(f"Task {task_id} not found after processing for COMPLETED state.")
                    return # Exit if task is gone
            
            # Publish final status update with the COMPLETED state AFTER lock release
            if completed_task_to_publish:
                await self._publish_status_update(completed_task_to_publish, True)
                # Publish artifacts if available, also after lock release
                for artifact in completed_task_to_publish.artifacts:
                     # _publish_artifact_update will acquire its own lock as needed
                    await self._publish_artifact_update(completed_task_to_publish, artifact)
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            logger.error(traceback.format_exc())
            
            async with self.lock: # Lock for FAILED state update due to general error
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    
                    # Update task status to FAILED
                    task.status = TaskStatus(
                        state=TaskState.FAILED,
                        message=Message(
                            role="agent",
                            parts=[{"type": "text", "text": f"Error processing financial task: {str(e)}"}]
                        ),
                        timestamp=datetime.now()
                    )
                    
                    logger.info(f"Updated task {task_id} status to FAILED")
                    self.tasks[task_id] = task
                    failed_task_to_publish_on_error = task # Prepare for publishing after lock release
                else:
                    logger.warning(f"Task {task_id} not found for FAILED update due to error: {e}")
                    return # Exit if task is gone

            # Publish final status update AFTER lock release
            if failed_task_to_publish_on_error:
                await self._publish_status_update(failed_task_to_publish_on_error, True)
    
    async def _publish_status_update(self, task: Task, final: bool) -> None:
        """
        Publish a status update for a task.

        Args:
            task: The task object
            final: Whether this is the final status update
        """
        task_id = task.id
        logger.debug(f"[_publish_status_update] Called for task {task_id}, final: {final}, state: {task.status.state}")
        status_event = TaskStatusUpdateEvent(
            id=task_id,
            status=task.status,
            final=final,
            metadata=task.metadata
        )
        logger.debug(f"[_publish_status_update] Created status_event for task {task_id}")

        logger.debug(f"[_publish_status_update] Attempting to publish to subscribers for task {task_id}")
        await self._publish_to_subscribers(task_id, status_event)
        logger.debug(f"[_publish_status_update] Finished publishing to subscribers for task {task_id}")

        logger.debug(f"[_publish_status_update] PRE-PUSH: About to attempt sending push notification for task {task_id}")
        await self._send_push_notification(task_id, status_event)
        logger.debug(f"[_publish_status_update] POST-PUSH: Finished sending push notification for task {task_id}")
        logger.info(f"Status event for task {task_id}: state={task.status.state}, message='...'")
    
    async def _publish_artifact_update(self, task: Task, artifact: Artifact) -> None:
        """
        Publish an artifact update to subscribers.
        
        Args:
            task: The task the artifact belongs to
            artifact: The artifact to publish
        """
        task_id = task.id
        
        # Extract artifact content for logging
        artifact_content = ""
        for part in artifact.parts:
            if hasattr(part, "type") and part.type == "text":
                artifact_content += part.text
            elif isinstance(part, dict) and part.get("type") == "text":
                artifact_content += part.get("text", "")
        
        logger.info(f"Publishing artifact for task {task_id}: {artifact.name}")
        logger.debug(f"Artifact content: '{artifact_content}'")
        
        # Create an artifact update event
        artifact_event = TaskArtifactUpdateEvent(
            id=task_id,
            artifact=artifact,
            metadata=task.metadata
        )
        
        # Publish to subscribers
        await self._publish_to_subscribers(task_id, artifact_event)
    
    async def _publish_to_subscribers(self, task_id: str, event: Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]) -> None:
        """
        Publish an event to all subscribers of a task.
        
        Args:
            task_id: ID of the task
            event: The event to publish
        """
        event_type = "status" if isinstance(event, TaskStatusUpdateEvent) else "artifact"
        logger.info(f"Publishing {event_type} event to subscribers for task {task_id}")
        
        async with self.lock:
            # Call registered callbacks
            if task_id in self.task_subscribers:
                subscribers = list(self.task_subscribers[task_id])
                
                logger.info(f"Found {len(subscribers)} subscribers for task {task_id}")
                for i, callback in enumerate(subscribers):
                    try:
                        logger.info(f"Calling subscriber {i+1}/{len(subscribers)} for task {task_id}")
                        callback(event)
                        logger.info(f"Successfully called subscriber {i+1} for task {task_id}")
                    except Exception as e:
                        logger.error(f"Error calling subscriber callback for task {task_id}: {e}")
                        logger.error(traceback.format_exc())
                
                # Clean up subscribers if final update
                if isinstance(event, TaskStatusUpdateEvent) and event.final:
                    logger.info(f"Final update for task {task_id}, removing subscribers")
                    self.task_subscribers.pop(task_id, None)
            else:
                logger.warning(f"No subscribers found for task {task_id}")
            
            # Send push notification if configured
            if task_id in self.push_notification_configs:
                logger.info(f"Sending push notification for task {task_id}")
                await self._send_push_notification(task_id, event)
    
    async def _send_push_notification(self, task_id: str, event: Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]) -> None:
        """
        Send a push notification for a task event.

        Args:
            task_id: The ID of the task
            event: The event to send (status or artifact update)
        """
        logger.debug(f"[_send_push_notification] Called for task {task_id}")
        push_config = None
        async with self.lock:
            push_config = self.push_notification_configs.get(task_id)

        if not push_config:
            logger.debug(f"[_send_push_notification] No push config found for task {task_id}, skipping.")
            return

        if not push_config.url:
            logger.warning(f"[_send_push_notification] Push config for task {task_id} has no URL, skipping.")
            return

        payload = event.model_dump_json()
        headers = {'Content-Type': 'application/json'}
        if push_config.headers:
            headers.update(push_config.headers)

        max_retries = 3
        base_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                logger.debug(f"[_send_push_notification] Attempt {attempt + 1}/{max_retries} to POST to {push_config.url} for task {task_id}")
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        push_config.url,
                        content=payload,
                        headers=headers,
                        timeout=10.0  # Added a timeout to the HTTP POST
                    )
                logger.debug(f"[_send_push_notification] POST attempt {attempt + 1} to {push_config.url} for task {task_id} completed with status {response.status_code}")
                response.raise_for_status()  # Raise an exception for bad status codes
                logger.info(f"Push notification sent successfully for task {task_id} to {push_config.url}, attempt {attempt + 1}")
                return  # Success
            except httpx.TimeoutException as e:
                logger.warning(f"[_send_push_notification] Timeout sending push notification for task {task_id} to {push_config.url}, attempt {attempt + 1}: {e}")
            except httpx.RequestError as e:
                logger.warning(f"[_send_push_notification] RequestError sending push notification for task {task_id} to {push_config.url}, attempt {attempt + 1}: {e}")
            except httpx.HTTPStatusError as e:
                logger.error(f"[_send_push_notification] HTTPStatusError sending push notification for task {task_id} to {push_config.url}, attempt {attempt + 1}: {e.response.status_code} - {e.response.text}")
                # Don't retry on client errors (4xx) typically, but server errors (5xx) might be retried
                if 400 <= e.response.status_code < 500:
                    logger.error(f"[_send_push_notification] Client error for task {task_id}, not retrying.")
                    break 
            except Exception as e:
                logger.error(f"[_send_push_notification] Unexpected error sending push notification for task {task_id} to {push_config.url}, attempt {attempt + 1}: {e}")
                logger.error(traceback.format_exc())

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"[_send_push_notification] Retrying push notification for task {task_id} in {delay}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"[_send_push_notification] Max retries reached for push notification to task {task_id} at {push_config.url}.")
        logger.debug(f"[_send_push_notification] Finished processing for task {task_id}") 