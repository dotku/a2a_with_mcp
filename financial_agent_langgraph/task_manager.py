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

from .agent import process_financial_task

# Configure more detailed logging
logger = logging.getLogger(__name__)

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
        await self._publish_status_update(task, False)
        
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
        Process a task using the financial agent.
        
        Args:
            task_id: ID of the task to process
        """
        logger.info(f"Processing task: {task_id}")
        async with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found for processing")
                return
            
            task = self.tasks[task_id]
            
            # Update task status to WORKING
            task.status = TaskStatus(
                state=TaskState.WORKING,
                timestamp=datetime.now()
            )
            
            logger.info(f"Updated task {task_id} status to WORKING")
            self.tasks[task_id] = task
        
        # Publish status update
        await self._publish_status_update(task, False)
        
        try:
            # Process the task using the financial agent
            # This is non-blocking
            logger.info(f"Calling process_financial_task for task {task_id}")
            updated_task = await asyncio.to_thread(process_financial_task, task)
            logger.info(f"Received result from process_financial_task for task {task_id}")
            logger.debug(f"Updated task details: {updated_task}")
            
            async with self.lock:
                self.tasks[task_id] = updated_task
                logger.info(f"Stored updated task {task_id} with status: {updated_task.status.state}")
            
            # Log artifacts before publishing
            if updated_task.artifacts:
                for i, artifact in enumerate(updated_task.artifacts):
                    artifact_content = ""
                    for part in artifact.parts:
                        if hasattr(part, "type") and part.type == "text":
                            artifact_content += part.text
                        elif isinstance(part, dict) and part.get("type") == "text":
                            artifact_content += part.get("text", "")
                    logger.info(f"Artifact {i+1}/{len(updated_task.artifacts)} - {artifact.name}: '{artifact_content[:100]}...'")
            
            # Publish the final status update
            logger.info(f"Publishing final status update for task {task_id}: {updated_task.status.state}")
            await self._publish_status_update(updated_task, True)
            
            # Publish any artifacts
            if updated_task.artifacts:
                logger.info(f"Publishing {len(updated_task.artifacts)} artifacts for task {task_id}")
                for artifact in updated_task.artifacts:
                    await self._publish_artifact_update(updated_task, artifact)
                    
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            logger.error(traceback.format_exc())
            
            async with self.lock:
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
                    
                    # Publish final status update
                    await self._publish_status_update(task, True)
    
    async def _publish_status_update(self, task: Task, final: bool) -> None:
        """
        Publish a task status update to subscribers.
        
        Args:
            task: The task to publish updates for
            final: Whether this is the final update for the task
        """
        task_id = task.id
        logger.info(f"Publishing status update for task {task_id}, state: {task.status.state}, final: {final}")
        
        # Create a status update event
        status_event = TaskStatusUpdateEvent(
            id=task_id,
            status=task.status,
            final=final,
            metadata=task.metadata
        )
        
        # Log the event
        message_text = ""
        if task.status.message:
            for part in task.status.message.parts:
                if hasattr(part, "type") and part.type == "text":
                    message_text += part.text
                elif isinstance(part, dict) and part.get("type") == "text":
                    message_text += part.get("text", "")
        
        logger.info(f"Status event for task {task_id}: state={task.status.state}, message='{message_text[:100]}...'")
        
        # Publish to subscribers
        await self._publish_to_subscribers(task_id, status_event)
    
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
        
        This is a placeholder for actual push notification implementation.
        
        Args:
            task_id: ID of the task
            event: The event to send
        """
        # This is a placeholder for actual push notification implementation
        # In a real implementation, this would make an HTTP request to the push notification URL
        config = self.push_notification_configs.get(task_id)
        if not config:
            logger.warning(f"No push notification config found for task {task_id}")
            return
        
        # Log the push notification for debugging
        logger.info(f"Would send push notification for task {task_id} to {config.url}")
        logger.debug(f"Push notification event: {event}")
        
        # In a real implementation, we would:
        # 1. Serialize the event to JSON
        # 2. Make an HTTP POST request to config.url
        # 3. Include the token in the Authorization header if provided 