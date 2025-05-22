#!/usr/bin/env python3
"""
Launcher script for the sentiment analysis agent server.
This makes it easier to start the agent server from the command line.

Usage:
  python start_agent_server.py

Environment Variables:
  SENTIMENT_AGENT_PORT - Port for the agent server (default: 10000)
  GOOGLE_API_KEY - API key for Google services (required)
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

if __name__ == "__main__":
    # Load environment variables from .env file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    # Try to load .env from current directory, then parent directory
    if os.path.exists(os.path.join(base_dir, '.env')):
        load_dotenv(os.path.join(base_dir, '.env'))
        print(f"Loaded environment from {os.path.join(base_dir, '.env')}")
    elif os.path.exists(os.path.join(parent_dir, '.env')):
        load_dotenv(os.path.join(parent_dir, '.env'))
        print(f"Loaded environment from {os.path.join(parent_dir, '.env')}")
    else:
        print("No .env file found. Using existing environment variables.")

    # Check if GOOGLE_API_KEY is set
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable must be set.")
        print("Please create a .env file with GOOGLE_API_KEY=your_api_key or set it in your environment.")
        sys.exit(1)
        
    # Get the port from the environment
    host = "0.0.0.0"
    port = int(os.environ.get("SENTIMENT_AGENT_PORT", "10000"))
    
    print(f"Starting sentiment analysis agent server on {host}:{port}...")
    
    # Run the server directly using the app from server.py
    # Import here to avoid circular imports
    from agents.sentiment_agent.server import app
    uvicorn.run(app, host=host, port=port, log_level="info") 