"""
Entry point for running the Financial Agent server.
"""
import os
import logging
import asyncio
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for running the Financial Agent server.
    """
    # Get port from environment, with fallback
    port = int(os.getenv("PORT", 8001))
    
    # Log server startup
    logger.info(f"Starting Financial Agent server on port {port}")
    
    # Run the server using uvicorn with absolute import path
    uvicorn.run(
        "agents.financial_agent.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )

if __name__ == "__main__":
    main() 