"""
Task Manager for Sentiment Analysis Agent.
Manages A2A tasks and interfaces with the CrewAI sentiment analysis agent.
"""
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional, Union, AsyncIterable
import uuid
import os
import sys
import asyncio

from sentiment_analysis_agent.common.types import (
    Task,
    TaskStatus,
    TaskState,
    Message,
    Part,
    TextPart,
    Artifact,
    TaskPushNotificationConfig,
    UnsupportedOperationError,
    TaskNotFoundError,
    TaskNotCancelableError,
)

# Import the underlying agent - updated to use the new module name
try:
    from sentiment_analysis_agent.agent_core.agent import SentimentAnalysisAgent
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import SentimentAnalysisAgent: {e}")
    logger.error("This might be due to missing dependencies or conflicts with local modules.")
    logger.error("Make sure crewai is installed: pip install crewai>=0.27.0")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_task_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentTaskManager:
    """Task manager for the Sentiment Analysis Agent."""
    
    def __init__(self):
        """Initialize the task manager."""
        logger.info("Initializing SentimentTaskManager")
        self.tasks: Dict[str, Task] = {}
        self.sentiment_agent = SentimentAnalysisAgent()
        logger.info("SentimentTaskManager initialized")
    
    async def send_task(self, params) -> Task:
        """
        Process a task/send request.
        
        Args:
            params: Parameters from the A2A request
            
        Returns:
            Task: The updated task
        """
        task_id = params.id
        session_id = params.sessionId or "default_session"
        
        logger.info(f"Processing task: {task_id}")
        
        # Extract the message text from the first part
        message = params.message
        message_text = ""
        
        for part in message.parts:
            if isinstance(part, TextPart):
                message_text = part.text
                break
        
        # Create and store initial task state
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        task = Task(
            id=task_id,
            sessionId=session_id,
            status=TaskStatus(
                state=TaskState.SUBMITTED,
                message=message,
                timestamp=timestamp
            ),
            artifacts=None,
            history=[message],  # Initialize history with the user message
            metadata=None
        )
        
        self.tasks[task_id] = task
        
        try:
            # Update status to working
            task.status = TaskStatus(
                state=TaskState.WORKING,
                message=None,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
            
            # Call the sentiment agent in a non-blocking way using asyncio.to_thread
            logger.info(f"Calling sentiment agent with query: {message_text}")
            response = await asyncio.to_thread(self.sentiment_agent.invoke, message_text, session_id)
            
            # Extract the raw string response
            if hasattr(response, 'raw'):
                response_text = response.raw
            else:
                response_text = str(response)
            
            # Create a response artifact
            artifact = Artifact(
                name="sentiment-analysis",
                description="Bitcoin sentiment analysis results",
                parts=[TextPart(type="text", text=response_text)],
                index=0,
                append=None,
                lastChunk=None,
                metadata=None
            )
            
            # Create a response message for history
            response_message = Message(
                role="assistant",
                parts=[TextPart(type="text", text=response_text)],
                metadata=None
            )
            
            # Update the task with completed status, the artifact, and history
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message=None,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
            task.artifacts = [artifact]
            
            # Add the response to history
            if task.history is None:
                task.history = [message, response_message]
            else:
                task.history.append(response_message)
            
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            logger.error(traceback.format_exc())
            
            # Create an error message
            error_message = Message(
                role="agent",
                parts=[TextPart(type="text", text=f"Error: {str(e)}")],
                metadata=None
            )
            
            # Update the task with failed status
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=error_message,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
            
            # Add the error message to history
            if task.history is None:
                task.history = [message, error_message]
            else:
                task.history.append(error_message)
        
        # Store the updated task
        self.tasks[task_id] = task
        
        return task
    
    async def get_task(self, params) -> Optional[Task]:
        """
        Get the current state of a task.
        
        Args:
            params: Parameters from the A2A request
            
        Returns:
            Task: The task object or None if not found
        """
        task_id = params.id
        logger.info(f"Getting task: {task_id}")
        
        if task_id not in self.tasks:
            logger.warning(f"Task not found: {task_id}")
            return None
        
        return self.tasks[task_id]
    
    async def cancel_task(self, params) -> Optional[Task]:
        """
        Cancel a task if possible.
        
        Args:
            params: Parameters from the A2A request
            
        Returns:
            Task: The updated task or None if not found
        """
        task_id = params.id
        logger.info(f"Canceling task: {task_id}")
        
        if task_id not in self.tasks:
            logger.warning(f"Task not found for cancellation: {task_id}")
            return None
        
        task = self.tasks[task_id]
        
        # Check if the task is in a final state
        if task.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
            logger.warning(f"Task {task_id} is already in final state: {task.status.state}")
            raise TaskNotCancelableError()
        
        # Update the task status to canceled
        task.status = TaskStatus(
            state=TaskState.CANCELED,
            message=None,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        # Store the updated task
        self.tasks[task_id] = task
        
        return task
    
    async def set_push_notification(self, params) -> TaskPushNotificationConfig:
        """
        Set push notification for a task.
        This agent does not support push notifications.
        
        Args:
            params: Parameters from the A2A request
            
        Returns:
            TaskPushNotificationConfig: The push notification config
            
        Raises:
            UnsupportedOperationError: This agent does not support push notifications
        """
        logger.warning("Push notifications are not supported")
        raise UnsupportedOperationError("Push notifications are not supported")
    
    async def get_push_notification(self, params) -> Optional[TaskPushNotificationConfig]:
        """
        Get push notification config for a task.
        This agent does not support push notifications.
        
        Args:
            params: Parameters from the A2A request
            
        Returns:
            TaskPushNotificationConfig: The push notification config
            
        Raises:
            UnsupportedOperationError: This agent does not support push notifications
        """
        logger.warning("Push notifications are not supported")
        raise UnsupportedOperationError("Push notifications are not supported")
    
    async def resubscribe_task(self, params, callback) -> None:
        """
        Resubscribe to a task for streaming updates.
        This agent does not support streaming.
        
        Args:
            params: Parameters from the A2A request
            callback: Function to call with updates
            
        Raises:
            UnsupportedOperationError: This agent does not support streaming
        """
        logger.warning("Streaming is not supported")
        raise UnsupportedOperationError("Streaming is not supported") 