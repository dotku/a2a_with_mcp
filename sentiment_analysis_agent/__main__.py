import os
import sys
import uvicorn
import logging
from pathlib import Path
from dotenv import load_dotenv

# Set up logging for the main entry point
logger = logging.getLogger("__main__")
logging.basicConfig(level=logging.INFO)

# Use relative import for the app from server.py
# Ensure server.py doesn't have conflicting top-level uvicorn.run() calls
from .server import app

if __name__ == "__main__":
    # Determine base and parent directory relative to this __main__.py file
    # __file__ is .../sentiment_analysis_agent/__main__.py
    # base_dir is .../sentiment_analysis_agent
    # parent_dir is .../ (workspace root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    # --- Environment Setup --- 
    # Load environment variables from .env file (look in agent dir, then workspace root)
    env_path_agent = os.path.join(base_dir, '.env')
    env_path_root = os.path.join(parent_dir, '.env')
    
    loaded_env = False
    if os.path.exists(env_path_agent):
        load_dotenv(env_path_agent)
        logger.info(f"Loaded environment from {env_path_agent}")
        loaded_env = True
    elif os.path.exists(env_path_root):
        load_dotenv(env_path_root)
        logger.info(f"Loaded environment from {env_path_root}")
        loaded_env = True
    else:
        logger.info("No .env file found in agent or root directory. Using existing environment variables.")
    
    # Check if GOOGLE_API_KEY is set
    if not os.environ.get("GOOGLE_API_KEY"):
        logger.error("Error: GOOGLE_API_KEY environment variable must be set.")
        logger.error("Please create a .env file with GOOGLE_API_KEY=your_api_key or set it in your environment.")
        sys.exit(1)
        
    # --- PYTHONPATH Setup (Important for module execution) ---
    # Ensure the 'python' directory (containing 'common') is in the Python path
    python_dir = os.path.join(parent_dir, "python") # parent_dir is workspace root
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
        logger.info(f"Added python directory ({python_dir}) to sys.path")

    # --- Server Startup --- 
    host = "0.0.0.0"
    port = int(os.environ.get("SENTIMENT_AGENT_PORT", "10000"))
    
    logger.info(f"Starting sentiment analysis agent server on {host}:{port}")
    logger.info(f"Current sys.path: {sys.path}") # Log path for debugging
    
    # Run uvicorn programmatically
    # No need to modify env['PYTHONPATH'] here, as sys.path is modified above
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.exception(f"Error starting server: {e}", exc_info=True)
        sys.exit(1) 