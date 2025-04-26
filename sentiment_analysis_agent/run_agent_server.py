#!/usr/bin/env python3
"""
Launcher script for the sentiment analysis agent server.
This script handles Python path setup to avoid import conflicts.

Usage:
  python run_agent_server.py

Environment Variables:
  SENTIMENT_AGENT_PORT - Port for the agent server (default: 10000)
  GOOGLE_API_KEY - API key for Google services (required)
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    # Check if GOOGLE_API_KEY is set
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable must be set.")
        sys.exit(1)
    
    # Get the absolute path to the sentiment analysis agent directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a new environment with PYTHONPATH set to include our parent directory
    # This ensures modules are found correctly
    env = os.environ.copy()
    parent_dir = os.path.dirname(base_dir)
    
    # If PYTHONPATH is already set, append to it; otherwise, create it
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{parent_dir}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = parent_dir
    
    # Set the port
    port = env.get("SENTIMENT_AGENT_PORT", "10000")
    
    # Define the command to run uvicorn directly
    cmd = [
        sys.executable, "-m", "uvicorn",
        "sentiment_analysis_agent.agent_server:app",
        "--host", "0.0.0.0",
        "--port", port,
        "--log-level", "info"
    ]
    
    print(f"Starting sentiment analysis agent server on port {port}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    
    # Run the server as a subprocess with the modified environment
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1) 