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
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

async def init_mcp_before_server():
    """Initialize MCP client before starting the server."""
    logger.info("Pre-initializing MCP client before server startup...")
    try:
        from agents.financial_agent.agent import ensure_mcp_initialized
        await ensure_mcp_initialized()
        logger.info("MCP client successfully pre-initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to pre-initialize MCP client: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """
    Main entry point for running the Financial Agent server.
    """
    # Get port from environment, with fallback
    port = int(os.getenv("PORT", 8001))
    
    # Log server startup
    logger.info(f"Starting Financial Agent server on port {port}")
    
    # Pre-initialize MCP client before starting server
    logger.info("Phase 1: Pre-initializing MCP client...")
    mcp_success = asyncio.run(init_mcp_before_server())
    
    if not mcp_success:
        logger.error("MCP pre-initialization failed. Server may not function properly.")
        # Continue anyway - the server can still start without MCP
    
    logger.info("Phase 2: Starting FastAPI server...")
    # Run the server using uvicorn with absolute import path
    uvicorn.run(
        "agents.financial_agent.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )

if __name__ == "__main__":
    main() 