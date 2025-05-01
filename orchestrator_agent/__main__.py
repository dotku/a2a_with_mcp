import uvicorn
import os
import sys
import logging

# Determine base and parent directory relative to this __main__.py file
base_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(base_dir)
python_dir = os.path.join(workspace_root, "python")

# --- PYTHONPATH Setup (Important for module execution) ---
# Ensure the 'python' directory (containing 'common') is in the Python path
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)
    # No logger configured yet, print instead
    print(f"Added python directory ({python_dir}) to sys.path") 

# Use relative import to get the app object from server.py
# This import should happen *after* sys.path is potentially modified
from .server import app, host, port

logger = logging.getLogger("__main__")

if __name__ == "__main__":
    # Set up basic logging for the main entry point
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Current sys.path: {sys.path}") # Log path for debugging
    
    # Run the server using the app object imported from server.py
    uvicorn.run(app, host=host, port=port) 