"""Client for interacting with the Visualization Agent."""

import json
import logging
import uuid
import requests
import os
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class VisualizationClient:
    """Client for the Visualization Agent."""

    def __init__(self, base_url: str = None):
        """
        Initialize the client.
        
        Args:
            base_url: The base URL of the visualization agent.
        """
        self.base_url = base_url or os.environ.get("VISUALIZATION_AGENT_URL", "http://localhost:8004")
        # Add a unique client ID to identify this UI instance
        self.client_id = f"ui-client-{uuid.uuid4()}"
        # Test the connection
        self._test_connection()
        
    def _test_connection(self) -> None:
        """Test the connection to the Visualization Agent."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"Successfully connected to Visualization Agent at {self.base_url}")
            else:
                logger.warning(f"Visualization Agent returned non-200 status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Visualization Agent: {e}")
    
    def generate_visualization_from_text(
        self,
        description: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a visualization from a natural language description.
        
        Args:
            description: Natural language description of the visualization to generate.
            session_id: Optional session ID to use for the request.
            
        Returns:
            A dictionary containing the visualization result.
        """
        if not session_id:
            session_id = f"ui-session-{uuid.uuid4()}"
            
        task_id = f"ui-task-{uuid.uuid4()}"
        
        # Create the JSON-RPC request with the raw text
        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "sessionId": session_id,
                "acceptedOutputModes": ["text", "artifact", "image/png"],
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": description
                        }
                    ]
                }
            }
        }
        
        # Add custom headers to identify this client
        headers = {
            "Content-Type": "application/json",
            "X-Client-ID": self.client_id
        }
        
        try:
            # Send the request to the Visualization Agent
            logger.info(f"Sending visualization text request to {self.base_url}")
            response = requests.post(
                self.base_url,
                json=request_data,
                headers=headers,
                timeout=30
            )
            
            return self._process_response(response, task_id, session_id)
        except Exception as e:
            logger.error(f"Error calling Visualization Agent: {e}")
            return {
                "status": "error",
                "message": f"Error calling Visualization Agent: {str(e)}",
                "task_id": task_id,
                "session_id": session_id
            }
            
    def generate_visualization(
        self, 
        data_type: str, 
        parameters: Dict[str, Any], 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a visualization using the Visualization Agent.
        
        Args:
            data_type: The type of visualization to generate (e.g., 'bar_chart', 'line_chart').
            parameters: Parameters for the visualization (e.g., x_axis_data, y_axis_data).
            session_id: Optional session ID to use for the request.
            
        Returns:
            A dictionary containing the visualization result.
        """
        if not session_id:
            session_id = f"ui-session-{uuid.uuid4()}"
            
        task_id = f"ui-task-{uuid.uuid4()}"
        
        # Format the data for the visualization agent
        visualization_request = {
            "plot_description": f"Generate a {data_type}",
            "data_json": json.dumps({
                "data_type": data_type,
                "parameters": parameters
            })
        }
        
        # Create the JSON-RPC request
        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "sessionId": session_id,
                "acceptedOutputModes": ["text", "artifact"],
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": json.dumps(visualization_request)
                        }
                    ]
                }
            }
        }
        
        # Add custom headers to identify this client
        headers = {
            "Content-Type": "application/json",
            "X-Client-ID": self.client_id
        }
        
        try:
            # Send the request to the Visualization Agent
            logger.info(f"Sending visualization request to {self.base_url}")
            response = requests.post(
                self.base_url,
                json=request_data,
                headers=headers,
                timeout=30
            )
            
            return self._process_response(response, task_id, session_id)
        except Exception as e:
            logger.error(f"Error calling Visualization Agent: {e}")
            return {
                "status": "error",
                "message": f"Error calling Visualization Agent: {str(e)}",
                "task_id": task_id,
                "session_id": session_id
            }
    
    def _process_response(self, response, task_id, session_id) -> Dict[str, Any]:
        """Process the response from the Visualization Agent."""
        # Handle the response
        if response.status_code == 200:
            try:
                result = response.json()
                logger.info(f"Received response from Visualization Agent: {json.dumps(result)[:200]}...")
                
                # Check if the task was completed successfully
                if (
                    "result" in result 
                    and isinstance(result["result"], dict)
                    and "status" in result["result"]
                    and isinstance(result["result"]["status"], dict)
                    and "state" in result["result"]["status"]
                    and result["result"]["status"]["state"] == "completed"
                ):
                    # Extract the visualization data
                    if "artifacts" in result["result"] and result["result"]["artifacts"]:
                        for artifact in result["result"]["artifacts"]:
                            if "parts" in artifact and artifact["parts"]:
                                for part in artifact["parts"]:
                                    if "file" in part and "bytes" in part["file"]:
                                        # Return the image data
                                        return {
                                            "status": "success",
                                            "image_data": {
                                                "name": part["file"].get("name", "visualization.png"),
                                                "mime_type": part["file"].get("mimeType", "image/png"),
                                                "data": part["file"]["bytes"]
                                            }
                                        }
                    
                    # If no image data was found, return a generic success
                    return {
                        "status": "success",
                        "message": "Visualization task completed successfully but no image data was found",
                        "task_id": task_id,
                        "session_id": session_id
                    }
                elif (
                    "result" in result 
                    and isinstance(result["result"], dict)
                    and "status" in result["result"]
                    and isinstance(result["result"]["status"], dict)
                    and "state" in result["result"]["status"]
                    and result["result"]["status"]["state"] in ["failed", "error"]
                ):
                    # Handle error state
                    error_message = "Unknown error occurred"
                    if (
                        "message" in result["result"]["status"]
                        and isinstance(result["result"]["status"]["message"], dict)
                        and "parts" in result["result"]["status"]["message"]
                        and result["result"]["status"]["message"]["parts"]
                        and "text" in result["result"]["status"]["message"]["parts"][0]
                    ):
                        error_message = result["result"]["status"]["message"]["parts"][0]["text"]
                        
                    return {
                        "status": "error",
                        "message": error_message,
                        "task_id": task_id,
                        "session_id": session_id
                    }
                else:
                    # Handle unexpected response format
                    return {
                        "status": "error",
                        "message": "Unexpected response format from Visualization Agent",
                        "raw_response": result,
                        "task_id": task_id,
                        "session_id": session_id
                    }
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "message": "Invalid JSON response from Visualization Agent",
                    "task_id": task_id,
                    "session_id": session_id
                }
        else:
            return {
                "status": "error",
                "message": f"Visualization Agent returned error status: {response.status_code}",
                "task_id": task_id,
                "session_id": session_id
            }

# Create a singleton instance
visualization_client = VisualizationClient() 