"""Visualization Agent Task Manager."""

import json
import logging
import traceback
from typing import AsyncIterable, Any, Dict
import datetime # Added missing import
import base64 # Added for encoding
import os # Added for path joining

from pydantic import BaseModel, Field

# Assuming agent and common types are available
# Adjust imports based on final project structure
import sys
import asyncio # Import needed for lock
from typing import Any # Import needed for fallback definition

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import from the local common module within visualization_agent
from common.server.task_manager import InMemoryTaskManager
from common.server import utils
from common.types import (
    Artifact,
    FileContent,
    FilePart,
    JSONRPCResponse,
    JSONRPCRequest,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    Task,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TextPart,
    Message,
    UnsupportedOperationError,
    InternalError
)

from agents.visualization_agent.agent import VisualizationAgent, PlotData # Import agent

logger = logging.getLogger(__name__)
# BasicConfig removed, assuming setup in server.py or main entry point
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Input Validation Model ---
class PlotRequestInput(BaseModel):
    plot_description: str = Field(..., description="Natural language description of the plot needed.")
    data_json: str = Field(..., description="Data for the plot, formatted as a JSON string.")


class AgentTaskManager(InMemoryTaskManager):
    """Visualization Agent Task Manager."""

    def __init__(self, agent: VisualizationAgent):
        super().__init__()
        self.agent = agent

    async def _stream_generator(
        self, request: SendTaskRequest
    ) -> AsyncIterable[SendTaskResponse]:
        raise NotImplementedError("Streaming not supported by CrewAI agent")

    async def on_send_task(
        self, request: SendTaskRequest | JSONRPCRequest
    ) -> SendTaskResponse | AsyncIterable[SendTaskResponse]:
        # Check if this is a streaming request
        if isinstance(request, SendTaskStreamingRequest):
            logger.warning("Streaming is not supported by the visualization agent")
            return JSONRPCResponse(
                id=request.id,
                error=UnsupportedOperationError(message="This agent does not support streaming (tasks/sendSubscribe)")
            )
        
        # Handle params as dict or object
        params = request.params
        accepted_modes = None
        
        # Check for push notification request
        push_notification = None
        if isinstance(params, dict):
            push_notification = params.get("pushNotification")
            accepted_modes = params.get("acceptedOutputModes")
        elif hasattr(params, "pushNotification"):
            push_notification = params.pushNotification
            accepted_modes = params.acceptedOutputModes if hasattr(params, "acceptedOutputModes") else None
        
        # Return error if push notifications are requested
        if push_notification:
            logger.warning("Push notifications are not supported by the visualization agent")
            return JSONRPCResponse(
                id=request.id,
                error=UnsupportedOperationError(message="This agent does not support push notifications")
            )
        
        if accepted_modes is not None and not utils.are_modalities_compatible(
            accepted_modes,
            VisualizationAgent.SUPPORTED_CONTENT_TYPES,
        ):
            logger.warning(
                "Unsupported output mode. Received %s, Support %s",
                accepted_modes,
                VisualizationAgent.SUPPORTED_CONTENT_TYPES,
            )
            return utils.new_incompatible_types_error(request.id)

        # Extract task_id for logging
        task_id = params.get("id") if isinstance(params, dict) else params.id
        logger.info(f"Processing task: {task_id}")
        
        await self.upsert_task(params)
        
        # Normal invocation
        return await self._invoke(request)

    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        # Streaming not supported by underlying agent
        return utils.new_method_not_found_error(request.id)

    async def _invoke(self, request: SendTaskRequest | JSONRPCRequest) -> SendTaskResponse | JSONRPCResponse:
        # Handle params as dict or object
        params = request.params
        task_id = None
        session_id = None
        message = None
        
        # Handle both dict and object cases for params
        if isinstance(params, dict):
            task_id = params.get("id")
            session_id = params.get("sessionId")
            message = params.get("message")
            push_notification = params.get("pushNotification")
        else:
            task_id = params.id
            session_id = params.sessionId
            message = params.message
            push_notification = getattr(params, "pushNotification", None)
            
        logger.info(f"Invoking task manager for task: {task_id} with session: {session_id}")
        
        # Strictly validate pushNotification if provided
        if push_notification:
            if not isinstance(push_notification, dict) and not hasattr(push_notification, "url"):
                logger.error(f"Invalid push notification format for task {task_id}")
                return utils.new_invalid_params_error(
                    request.id, 
                    "pushNotification must have a 'url' field"
                )
                
            # Validate URL format if using dict
            if isinstance(push_notification, dict) and not push_notification.get("url"):
                logger.error(f"Missing required URL in push notification for task {task_id}")
                return utils.new_invalid_params_error(
                    request.id, 
                    "pushNotification.url is required"
                )

        # Initialize plot_id and error_message variables
        plot_id = None
        error_message = None

        try:
            # Extract user input (plot description and data)
            user_input = self._extract_text_from_message(message)
                
            if not user_input:
                logger.error(f"No text input found in message for task {task_id}")
                error_message = "No text input found in message"
                return utils.new_invalid_params_error(
                    request.id,
                    error_message
                )
                
            # Parse the user input as JSON or as direct text
            try:
                # Check if input is direct text first (from UI) or JSON (from Orchestrator)
                if user_input.strip().startswith('{'):
                    # Try to parse as JSON
                    request_data = json.loads(user_input)
                    
                    plot_description = None
                    data_json = None
                    
                    # Check if this is coming from the orchestrator with specific format
                    if "plot_description" in request_data and "data_json" in request_data:
                        plot_description = request_data.get("plot_description")
                        # The data_json might be a string or already parsed JSON
                        data_json_value = request_data.get("data_json")
                        if isinstance(data_json_value, str):
                            data_json = data_json_value
                        else:
                            data_json = json.dumps(data_json_value)
                    # Or if it matches the expected parameters directly
                    elif "data_type" in request_data and "parameters" in request_data:
                        params = request_data.get("parameters", {})
                        data_type = request_data.get("data_type", "bar_chart")
                        
                        # Extract params for visualization
                        x_axis_data = params.get("x_axis_data", [])
                        y_axis_data = params.get("y_axis_data", [])
                        x_axis_label = params.get("x_axis_label", "X-Axis")
                        y_axis_label = params.get("y_axis_label", "Y-Axis")
                        title = params.get("title", "Visualization")
                        
                        # Format plot description
                        plot_description = f"Generate a {data_type} titled '{title}' with {x_axis_label} and {y_axis_label}"
                        
                        # Create data json
                        data_json = json.dumps({
                            "labels": x_axis_data,
                            "values": y_axis_data,
                            "xlabel": x_axis_label,
                            "ylabel": y_axis_label,
                            "title": title
                        })
                else:
                    # Handle direct text input from UI
                    # Parse the free-form text to extract visualization requirements
                    logger.info(f"Processing direct text input: {user_input[:100]}...")
                    
                    # Very simple parsing logic for demonstration
                    lines = user_input.strip().split('\n')
                    plot_description = lines[0] if lines else "Generate a visualization"
                    
                    # Extract chart type (default to bar chart)
                    chart_type = "bar_chart"
                    if "bar graph" in user_input.lower() or "bar chart" in user_input.lower():
                        chart_type = "bar_chart"
                    elif "line graph" in user_input.lower() or "line chart" in user_input.lower():
                        chart_type = "line_chart"
                    elif "pie chart" in user_input.lower():
                        chart_type = "pie_chart"
                    
                    # Extract labels and values
                    labels = []
                    values = []
                    
                    # Try to extract numbers from the input first
                    import re
                    potential_values_str = re.findall(r'\d+\.?\d*', user_input)
                    extracted_values = []
                    if potential_values_str:
                        try:
                            extracted_values = [float(v) for v in potential_values_str]
                            logger.info(f"Extracted potential numerical values: {extracted_values}")
                        except ValueError:
                            logger.warning("Could not convert some extracted strings to float for values.")
                            extracted_values = []

                    # Try to extract textual labels
                    extracted_labels = []
                    # Pattern 1: "x axis as A B C D", "labels are X Y Z", "categories: P Q R"
                    label_pattern_match = re.search(r'(?:x-?axis|labels|categories|names).*(?:as|are|:|is)\s*([A-Za-z0-9\s,]+)', user_input, re.IGNORECASE)
                    if label_pattern_match:
                        labels_str = label_pattern_match.group(1)
                        # Split by common delimiters and filter
                        extracted_labels = [name.strip() for name in re.split(r'[\s,]+', labels_str) if name.strip() and not name.lower() in ['and', 'with']]
                        logger.info(f"Extracted textual labels from pattern: {extracted_labels}")
                    
                    # Assign to final labels and values based on what was extracted
                    labels = []
                    values = []

                    if extracted_values and extracted_labels:
                        # Both provided, try to match lengths or prioritize
                        if len(extracted_values) == len(extracted_labels):
                            values = extracted_values
                            labels = extracted_labels
                            logger.info("Using extracted values and extracted labels directly (matched length).")
                        else:
                            # Length mismatch - a common scenario. Prioritize values, and truncate/pad labels.
                            values = extracted_values
                            if len(extracted_labels) > len(extracted_values):
                                labels = extracted_labels[:len(extracted_values)]
                                logger.warning(f"Label/Value length mismatch. Using {len(values)} values and truncating labels to: {labels}")
                            else: # len(extracted_labels) < len(extracted_values)
                                labels = extracted_labels + [f"Category {i+1}" for i in range(len(extracted_labels), len(extracted_values))]
                                logger.warning(f"Label/Value length mismatch. Using {len(values)} values and padding labels to: {labels}")
                    elif extracted_values: # Only values extracted
                        values = extracted_values
                        labels = [f"Category {i+1}" for i in range(len(values))]
                        logger.info(f"Only values extracted. Generated default labels: {labels}")
                    elif extracted_labels: # Only labels extracted
                        labels = extracted_labels
                        values = list(range(1, len(labels) + 1))
                        logger.info(f"Only labels extracted. Generated default values: {values}")
                    else:
                        # No direct values or specific label patterns found.
                        # Try the more general name/category extraction or line-by-line.
                        logger.debug("No direct values or specific label patterns found. Trying broader name extraction or line-by-line.")
                        
                        # Broader name extraction (less specific than the pattern above)
                        # Fallback: if no specific pattern, look for capitalized words or sequences as potential labels
                        temp_fallback_labels = []
                        if len(user_input.split()) > 3: # Heuristic
                            capitalized_words = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', user_input)
                            if capitalized_words:
                                temp_fallback_labels = capitalized_words
                            else:
                                non_command_words = []
                                excluded_keywords = {"generate", "create", "plot", "bar", "line", "pie", "chart", "graph", "visualization", "show", "showing", "with", "and", "for", "data", "values", "labels", "axis", "as", "is", "are", "like"}
                                potential_label_words = [word for word in user_input.split() if word.lower() not in excluded_keywords and not word.isdigit() and len(word) > 1]
                                if len(potential_label_words) >= 2: # Need at least two potential labels
                                    # Check if they are clumped together or part of a list-like phrase
                                    # This part can be made more sophisticated
                                    temp_fallback_labels = potential_label_words # Use them directly as labels

                        if temp_fallback_labels:
                            labels = temp_fallback_labels
                            values = list(range(1, len(labels) + 1))
                            logger.info(f"Extracted potential labels using broad fallback: {labels}, assigned default values.")
                        else:
                            # Last resort before hardcoded default: try label:value parsing
                            logger.debug("Trying original line-by-line parsing for label:value as a last resort before defaults.")
                            for line in lines:
                                if ":" in line and any(c.isdigit() for c in line.split(":",1)[1]): # check if there's a digit after colon
                                    parts = line.split(":", 1)
                                    if len(parts) == 2:
                                        label_part = parts[0].strip()
                                        value_part_str = parts[1].strip()
                                        try:
                                            value = float(value_part_str)
                                            labels.append(label_part)
                                            values.append(value)
                                        except ValueError:
                                            pass # Could not convert value part to float
                            if labels and values:
                                logger.info(f"Extracted labels and values from line-by-line parsing: {labels}, {values}")

                    # If no values AND no labels were extracted by any method, use some defaults
                    if not labels or not values:
                        labels = ["Category A", "Category B", "Category C"]
                        values = [100, 200, 300]
                    
                    # Extract title if mentioned
                    title = "Visualization"
                    for line in lines:
                        if "title" in line.lower() and ":" in line:
                            title = line.split(":", 1)[1].strip().strip('"').strip("'")
                            break
                    
                    # Extract axis labels if mentioned
                    x_axis_label = "X-Axis"
                    y_axis_label = "Y-Axis"
                    
                    if "x-axis" in user_input.lower() or "x axis" in user_input.lower():
                        for line in lines:
                            if ("x-axis" in line.lower() or "x axis" in line.lower()) and ":" in line:
                                x_axis_label = line.split(":", 1)[1].strip().strip('"').strip("'")
                                break
                    
                    if "y-axis" in user_input.lower() or "y axis" in user_input.lower():
                        for line in lines:
                            if ("y-axis" in line.lower() or "y axis" in line.lower()) and ":" in line:
                                y_axis_label = line.split(":", 1)[1].strip().strip('"').strip("'")
                                break
                    
                    # Create data json
                    data_json = json.dumps({
                        "labels": labels,
                        "values": values,
                        "xlabel": x_axis_label,
                        "ylabel": y_axis_label,
                        "title": title,
                        "x_axis_data": labels,
                        "y_axis_data": values,
                        "x_axis_label": x_axis_label,
                        "y_axis_label": y_axis_label
                    })
                    
                    # Update plot_description to be more specific
                    plot_description = f"Generate a {chart_type} showing {y_axis_label} for different {x_axis_label}"
                    
                if not plot_description or not data_json:
                    logger.error(f"Missing required parameters for task {task_id}")
                    error_message = "Missing required parameters: plot_description and data_json"
                    return utils.new_invalid_params_error(
                        request.id,
                        error_message
                    )
                    
                logger.info(f"Parsed input for task {task_id}: desc=\"{plot_description[:50]}...\", data=\"{data_json[:50]}...\"")
                
                # Update task status to working
                async with self.lock:
                    task = self.tasks.get(task_id)
                    if task:
                        task.status = TaskStatus(state=TaskState.WORKING)
                        logger.info(f"Updating task {task_id} status to WORKING")
                
                # Invoke the agent with the parsed parameters
                plot_id = self.agent.invoke(
                    plot_description=plot_description,
                    data_json=data_json,
                    session_id=session_id
                )
                
                if not plot_id:
                    logger.error(f"Agent failed to generate a plot_id for task {task_id}")
                    return utils.new_internal_error(
                        request.id,
                        "Failed to generate visualization"
                    )
                
                logger.info(f"Generated plot with ID {plot_id} for task {task_id}")

                # Retrieve the plot data (which includes base64 bytes) from the agent's cache
                image_bytes_data = None
                plot_data_obj = self.agent.get_plot_data(session_id, plot_id)
                if plot_data_obj and plot_data_obj.bytes:
                    image_bytes_data = plot_data_obj.bytes
                    logger.debug(f"Successfully retrieved base64 encoded image data for plot {plot_id} from cache.")
                else:
                    logger.error(f"Could not retrieve plot data or bytes for plot {plot_id} from cache. Artifact will not contain image bytes.")
                    # Potentially set task to failed or return an error if bytes are critical

                # Construct a simple text message for the task status
                image_url = f"/plots/{plot_id}.png"
                simple_response_text = f"Visualization generated. You can find it with ID: {plot_id} at {image_url}"
                status_update_message = Message(
                    role="assistant",
                    parts=[
                        TextPart(type="text", text=simple_response_text)
                    ]
                )

                # Create the artifact using FilePart and FileContent as per model definitions
                artifact = Artifact(
                    name="visualization_plot",
                    description=plot_description,
                    parts=[
                        FilePart(
                            type="file",
                            file=FileContent(
                                name=f"{plot_id}.png",
                                mimeType="image/png",
                                bytes=image_bytes_data, # Send the bytes
                                uri=None # Set URI to None if bytes are present
                            )
                        )
                    ]
                )
                logger.debug(f"Artifact created for task {task_id} with plot_id {plot_id}")
                
                async with self.lock:
                    task = self.tasks.get(task_id)
                    if task:
                        # Update task with response
                        task.status = TaskStatus(
                            state=TaskState.COMPLETED,
                            message=status_update_message, # Assign the text-only message
                            timestamp=datetime.datetime.now().isoformat() # Corrected usage
                        )
                        # Add the artifact
                        task.artifacts = [artifact] # Assign the correctly structured artifact
                        # Remove history tracking for this agent
                        if hasattr(task, 'history'):
                            task.history = None
                        logger.info(f"Updated task {task_id} with completed status and artifact")
                        # Send push notification if configured
                        if push_notification:
                            # Create status update event
                            status_event = TaskStatusUpdateEvent(
                                id=task_id,
                                status=task.status,
                                final=True
                            )
                            await self._send_push_notification(task_id, status_event)
                            # Create artifact update event
                            artifact_event = TaskArtifactUpdateEvent(
                                id=task_id,
                                artifact=artifact
                            )
                            await self._send_push_notification(task_id, artifact_event)
                
                # Return the task as the result
                return SendTaskResponse(
                    id=request.id,
                    result=task
                )
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for task {task_id}: {e}")
                return utils.new_parse_error(
                    request.id,
                    f"Error parsing JSON input: {str(e)}"
                )
            
            except Exception as e:
                logger.error(f"Error processing user input for task {task_id}: {e}")
                logger.exception("Detailed exception:")
                return utils.new_internal_error(
                    request.id,
                    f"Error processing visualization request: {str(e)}"
                )
                
        except Exception as e:
            logger.error(f"Unhandled error in _invoke for task {task_id}: {e}")
            logger.exception("Detailed exception:")
            return utils.new_internal_error(
                request.id,
                f"Internal error: {str(e)}"
            )

    def _extract_text_from_message(self, message) -> str:
        """Extract text content from message in different formats."""
        if not message:
            return ""
            
        # Handle Dict format
        if isinstance(message, dict):
            if "parts" in message:
                for part in message["parts"]:
                    if isinstance(part, dict) and "text" in part:
                        return part["text"]
                    elif isinstance(part, dict) and "type" in part and part["type"] == "text" and "text" in part:
                        return part["text"]
            # Direct message text (for simple UI clients)
            elif "text" in message:
                return message["text"]
        # Handle Object format
        elif hasattr(message, "parts") and message.parts:
            for part in message.parts:
                if hasattr(part, "text") and part.text:
                    return part.text
                elif hasattr(part, "type") and part.type == "text" and hasattr(part, "text"):
                    return part.text
        # Direct string input
        elif isinstance(message, str):
            return message
                    
        return ""

    # Ensure _update_store exists or use base implementation
    async def _update_store(
      self, task_id: str, status: TaskStatus, artifacts: list[Artifact] | None
    ) -> Task:
        async with self.lock:
            try:
                # Assume self.tasks is the dict holding Task objects
                task = self.tasks[task_id]
            except KeyError as exc:
                logger.error(f"Task {task_id} not found for updating the task")
                raise ValueError(f"Task {task_id} not found") from exc

            task.status = status

            # Append message to history if message tracking exists
            if status.message is not None and hasattr(self, 'task_messages') and task_id in self.task_messages:
                 self.task_messages[task_id].append(status.message)
                 task.history = self.task_messages[task_id] # Assuming history is tracked this way

            if artifacts is not None:
                if task.artifacts is None:
                    task.artifacts = []
                task.artifacts.extend(artifacts)

            # Update the task in the main dictionary
            self.tasks[task_id] = task
            return task 

    # Add an alias method to handle the tasks/send JSON-RPC method
    async def on_tasks_send(
        self, request: JSONRPCRequest
    ) -> SendTaskResponse | AsyncIterable[SendTaskResponse]:
        """Alias for on_send_task to handle method name mismatch."""
        logger.info(f"Received request via tasks/send, delegating to on_send_task")
        return await self.on_send_task(request) 

    async def upsert_task(self, params):
        """
        Create or update a task with the given params.
        Handles both dictionary and TaskSendParams object formats.
        """
        # Create a Task object that will be returned
        task_id = None
        session_id = None
        message = None
        # Handle both dict and object cases for params
        if isinstance(params, dict):
            task_id = params.get("id")
            session_id = params.get("sessionId")
            message = params.get("message")
        else:
            task_id = params.id
            session_id = params.sessionId
            message = params.message
        if not task_id:
            raise ValueError("Task ID is required")
        async with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus(state=TaskState.WORKING)
                # Remove history tracking for this agent
                if hasattr(task, 'history'):
                    task.history = None
            else:
                self.tasks[task_id] = Task(
                    id=task_id,
                    status=TaskStatus(state=TaskState.WORKING),
                    sessionId=session_id,
                    # Do not set history
                )
        return self.tasks[task_id] 

    async def _send_push_notification(self, task_id: str, event):
        """
        Send a push notification to the configured URL for a task event.
        
        Makes an HTTP POST request to the push notification URL with retry logic.
        
        Args:
            task_id: ID of the task
            event: The event to send
        """
        import httpx
        import json
        import asyncio
        
        # Get push notification config for the task
        async with self.lock:
            task = self.tasks.get(task_id)
            if not task or not task.push_notification:
                logger.warning(f"No push notification config found for task {task_id}")
                return
            
            config = task.push_notification
        
        # Validate the push notification config
        if not config.url:
            logger.error(f"Invalid push notification config for task {task_id}: missing URL")
            return
        
        # Serialize the event to JSON
        try:
            # Convert the event to a dictionary
            event_dict = event.model_dump(exclude_none=True) if hasattr(event, "model_dump") else event
            # Serialize to JSON
            event_json = json.dumps(event_dict)
        except Exception as e:
            logger.error(f"Error serializing push notification event for task {task_id}: {e}")
            return
        
        # Set up headers
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add authorization token if provided
        if config.token:
            headers["Authorization"] = f"Bearer {config.token}"
        
        # Initialize variables for retry logic
        max_retries = 3
        retry_count = 0
        base_delay = 1  # Base delay in seconds
        
        # Attempt to send the notification with retry logic
        while retry_count < max_retries:
            try:
                logger.info(f"Sending push notification for task {task_id} to {config.url} (attempt {retry_count + 1}/{max_retries})")
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        config.url,
                        headers=headers,
                        content=event_json
                    )
                    
                    if response.status_code >= 200 and response.status_code < 300:
                        logger.info(f"Successfully sent push notification for task {task_id}, status code: {response.status_code}")
                        return
                    else:
                        logger.warning(f"Push notification request failed for task {task_id} with status code {response.status_code}: {response.text}")
                
            except Exception as e:
                logger.error(f"Error sending push notification for task {task_id}: {str(e)}")
            
            # Increment retry count
            retry_count += 1
            
            # If we've reached max retries, log failure and return
            if retry_count >= max_retries:
                logger.error(f"Failed to send push notification for task {task_id} after {max_retries} attempts")
                return
            
            # Calculate exponential backoff delay
            delay = base_delay * (2 ** (retry_count - 1))
            logger.info(f"Retrying push notification for task {task_id} in {delay} seconds...")
            await asyncio.sleep(delay) 