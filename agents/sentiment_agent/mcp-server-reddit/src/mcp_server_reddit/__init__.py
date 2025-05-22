from .server import serve
import logging

logger = logging.getLogger(__name__)

def main():
    """MCP Reddit Server - Reddit API functionality for MCP"""
    import argparse
    import asyncio
    import sys

    parser = argparse.ArgumentParser(
        description="give a model the ability to access Reddit public API"
    )
    parser.add_argument('--port', type=int, default=10101,
                      help='Port for the HTTP server (default: 10101)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')

    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logger.info(f"Starting MCP Reddit Server on port {args.port}")
    
    try:
        asyncio.run(serve(port=args.port))
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
