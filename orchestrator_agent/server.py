import json
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional, AsyncGenerator

# Remove external path dependency
# sys.path.append('/home/anshul/Desktop/A2A/A2A/samples/python')

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from mcp.server.fastmcp import FastMCP

# Use local imports instead of external path
from common.server import A2AServer
from common.types import (
    AgentCard, 
    AgentCapabilities, 
    AgentSkill
)
# Remove this import and use the one from task_manager
# from common.utils.push_notification_auth import PushNotificationSenderAuth

from agent import process_request
from task_manager import OrchestratorTaskManager, PushNotificationSenderAuth
from mcp_server import mcp  # Import mcp from mcp_server.py instead of agent.py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define agent capabilities and skills
capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
skills = [
    AgentSkill(
        id="task_decomposition",
        name="Task Decomposition",
        description="Ability to break down complex tasks into manageable subtasks"
    ),
    AgentSkill(
        id="agent_delegation",
        name="Agent Delegation",
        description="Ability to assign tasks to appropriate specialized agents"
    ),
    AgentSkill(
        id="workflow_management",
        name="Workflow Management",
        description="Ability to coordinate and track progress across multiple agents"
    ),
    AgentSkill(
        id="result_integration",
        name="Result Integration",
        description="Ability to compile and integrate results from multiple agents"
    )
]

# Create FastAPI app
app = FastAPI(title="Orchestrator Agent", description="A2A-compliant Orchestrator Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up push notification sender auth
notification_sender_auth = PushNotificationSenderAuth()
notification_sender_auth.generate_jwk()

def create_server(host="localhost", port=8000):
    """Create and configure the A2A server"""
    
    # Create agent card
    agent_card = AgentCard(
        name="Orchestrator Agent",
        description="Agent that breaks down user tasks and delegates them to specialized agents",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        capabilities=capabilities,
        skills=skills,
        defaultInputModes=["text", "text/plain"],
        defaultOutputModes=["text", "text/plain"],
    )
    
    # Create task manager
    task_manager = OrchestratorTaskManager(notification_sender_auth)
    
    # Create A2A server
    server = A2AServer(
        agent_card=agent_card,
        task_manager=task_manager,
        host=host,
        port=port,
    )
    
    # Add JWKs endpoint for authentication
    server.app.add_route(
        "/.well-known/jwks.json",
        notification_sender_auth.handle_jwks_endpoint,
        methods=["GET"]
    )
    
    return server

# Get port from environment variable or use default
port = int(os.environ.get("PORT", 8000))
host = os.environ.get("HOST", "0.0.0.0")

# Create the A2A server
server = create_server(host, port)

# Mount the MCP server to the FastAPI app
app.mount("/mcp", mcp.sse_app())

# Mount the A2A server to our FastAPI app
app.mount("/", server.app)

@app.get("/.well-known/agent.json")
async def agent_manifest():
    """Return the agent manifest file."""
    try:
        with open(os.path.join(os.path.dirname(__file__), ".well-known/agent.json"), "r") as f:
            return Response(content=f.read(), media_type="application/json")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Agent manifest not found")

if __name__ == "__main__":
    logger.info(f"Starting server on {host}:{port}")
    
    # Run the server
    uvicorn.run(app, host=host, port=port) 