import uvicorn
import os
from agents.visualization_agent.server import app # Import the FastAPI app

if __name__ == "__main__":
    port = int(os.getenv("VISUALIZATION_AGENT_PORT", "8004"))
    host = os.getenv("VISUALIZATION_AGENT_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port) 