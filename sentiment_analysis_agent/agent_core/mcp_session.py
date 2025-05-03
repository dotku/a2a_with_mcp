"""Simplified MCP Server session handler."""

import os
import sys
import json
import logging
import uuid
import socket
import requests
import subprocess
import time
import threading
import queue

logger = logging.getLogger(__name__)

class SimpleMCPSession:
    """A simplified context manager for MCP server sessions."""
    
    def __init__(self, port=10101):
        """Initialize an MCP session with the given parameters"""
        self.port = port
        self.process = None
        self.server_ready = False
        self.max_retries = 3
        
    def __enter__(self):
        """Start the MCP server process when entering the context"""
        try:
            # Check if port is already in use
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', self.port))
                sock.close()
                
                if result == 0:  # Port is open/in use
                    # Try to make a health check request - maybe it's already our server
                    try:
                        response = requests.get(f"http://localhost:{self.port}/health", timeout=1)
                        if response.status_code == 200:
                            logger.info(f"MCP server already running on port {self.port}")
                            self.server_ready = True
                            # In this case, we don't start a new server process
                            return self
                    except requests.RequestException:
                        # If health check fails, it's probably another application
                        raise RuntimeError(f"Port {self.port} is already in use by another application")
            except socket.error:
                # Socket creation failed, unlikely but handle it
                logger.warning(f"Failed to check if port {self.port} is in use")
            
            # Get paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.normpath(os.path.join(current_dir, "..", "mcp-server-reddit", "src"))
            
            # Setup environment with proper PYTHONPATH
            env = os.environ.copy()
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = src_dir
            
            # Log what we're about to do
            logger.info(f"Starting MCP server from {src_dir}")
            logger.info(f"PYTHONPATH: {env.get('PYTHONPATH')}")
            
            # Start the server directly as a Python module
            self.process = subprocess.Popen(
                ["python", "-m", "mcp_server_reddit"], 
                cwd=src_dir,  # Run from the src directory for proper module imports
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Start threads to capture output for debugging
            def log_output(stream, prefix):
                for line in iter(stream.readline, ''):
                    if line.strip():
                        logger.debug(f"MCP {prefix}: {line.strip()}")
            
            stdout_thread = threading.Thread(target=log_output, args=(self.process.stdout, "stdout"))
            stderr_thread = threading.Thread(target=log_output, args=(self.process.stderr, "stderr"))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for the server to start
            logger.info(f"Waiting up to 30 seconds for MCP server to start...")
            start_time = time.time()
            
            while time.time() - start_time < 30:
                # Check if process has exited prematurely
                if self.process.poll() is not None:
                    raise RuntimeError(f"MCP server process exited with code {self.process.returncode}")
                
                # Check if server is responding
                try:
                    response = requests.get(f"http://localhost:{self.port}/health", timeout=1)
                    if response.status_code == 200:
                        logger.info("MCP server is up and running!")
                        self.server_ready = True
                        break
                except requests.RequestException:
                    # Expected while server is still starting
                    pass
                
                # Wait before trying again
                time.sleep(1)
            
            if not self.server_ready:
                raise RuntimeError(f"MCP server did not become ready within 30 seconds")
            
            return self
        
        except Exception as e:
            logger.exception(f"Error starting MCP server: {e}")
            if self.process:
                self.process.terminate()
                self.process = None
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Terminate the MCP server process when exiting the context"""
        if self.process:
            logger.info("Shutting down MCP server")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("MCP server did not terminate gracefully, killing process")
                self.process.kill()
                self.process.wait()
            logger.info("MCP server shutdown complete")
    
    def call(self, tool_name, arguments, retry_count=0):
        """Call an MCP tool and return the result"""
        try:
            if not self.server_ready:
                logger.error("MCP server is not ready")
                return json.dumps({"error": "MCP server is not ready"})
            
            # Create the JSON-RPC request
            request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "call_tool",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # Send the request
            logger.info(f"Calling MCP tool '{tool_name}' with arguments: {json.dumps(arguments)}")
            response = requests.post(
                f"http://localhost:{self.port}",
                json=request,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            # Process the response
            if response.status_code == 200:
                result = response.json()
                
                # Check for JSON-RPC errors
                if "error" in result:
                    logger.error(f"MCP server returned error: {result['error']}")
                    return json.dumps({"error": result["error"].get("message", "Unknown error")})
                
                # Return the result
                if "result" in result:
                    # Convert TextContent objects to a list of data
                    if isinstance(result["result"], list):
                        items = []
                        for item in result["result"]:
                            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                                try:
                                    # Try to parse each text item as JSON
                                    items.append(json.loads(item["text"]))
                                except json.JSONDecodeError:
                                    # If not valid JSON, include as raw text
                                    items.append({"text": item["text"]})
                        return json.dumps(items)
                    
                    # Return raw result if not a list of text items
                    return json.dumps(result["result"])
                
                # No result key found
                return json.dumps({"error": "No result in response"})
            else:
                logger.error(f"MCP server returned status code {response.status_code}: {response.text}")
                
                # Retry for server errors
                if retry_count < self.max_retries and 500 <= response.status_code < 600:
                    retry_count += 1
                    wait_time = retry_count * 2  # Exponential backoff
                    logger.warning(f"Retrying request (attempt {retry_count}/{self.max_retries}) after {wait_time}s")
                    time.sleep(wait_time)
                    return self.call(tool_name, arguments, retry_count)
                
                return json.dumps({"error": f"MCP server returned status code {response.status_code}"})
            
        except requests.RequestException as e:
            logger.exception(f"Request error calling MCP tool: {e}")
            
            # Retry for connection errors
            if retry_count < self.max_retries:
                retry_count += 1
                wait_time = retry_count * 2  # Exponential backoff
                logger.warning(f"Retrying request (attempt {retry_count}/{self.max_retries}) after {wait_time}s")
                time.sleep(wait_time)
                return self.call(tool_name, arguments, retry_count)
            
            return json.dumps({"error": f"Failed to connect to MCP server: {str(e)}"})
            
        except Exception as e:
            logger.exception(f"Error calling MCP tool: {e}")
            return json.dumps({"error": f"Error calling MCP tool: {str(e)}"}) 