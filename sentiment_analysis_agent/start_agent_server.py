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

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the run_server function from the task_manager module
from sentiment_analysis_agent.crewai.task_manager import run_server

if __name__ == "__main__":
    # Check if GOOGLE_API_KEY is set
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable must be set.")
        sys.exit(1)
        
    # Run the server
    print("Starting sentiment analysis agent server...")
    run_server() 