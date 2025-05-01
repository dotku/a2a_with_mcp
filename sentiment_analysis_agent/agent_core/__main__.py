"""This file serves as the main entry point for the application.

It initializes the A2A server, defines the agent's capabilities,
and starts the server to handle incoming requests.
"""

import os
import sys
# Add parent directory to Python path so we can import 'common'
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from agent import SentimentAnalysisAgent
import click
from common.server import A2AServer
from common.types import AgentCapabilities, AgentCard, AgentSkill, MissingAPIKeyError
import logging
from task_manager import AgentTaskManager
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10001)
def main(host, port):
  """Entry point for the A2A + CrewAI Bitcoin sentiment analysis sample."""
  try:
    if not os.getenv("GOOGLE_API_KEY"):
        raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")

    capabilities = AgentCapabilities(streaming=False)
    skill = AgentSkill(
        id="sentiment_analyzer",
        name="Bitcoin Sentiment Analyzer",
        description=(
            "Analyze Reddit data to determine the current sentiment around Bitcoin. "
            "Understand community feelings and identify key discussion topics in "
            "the cryptocurrency market."
        ),
        tags=["sentiment analysis", "bitcoin", "crypto", "reddit"],
        examples=["What's the current sentiment around Bitcoin?", "Analyze the Bitcoin market sentiment", "What are people saying about Bitcoin on Reddit?"],
    )

    agent_card = AgentCard(
        name="Bitcoin Sentiment Analysis Agent",
        description=(
            "Analyze Reddit data to determine the current sentiment around Bitcoin. "
            "Monitor community mood and identify trending topics in Bitcoin discussions "
            "to provide valuable market insights."
        ),
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text", "text/plain"],
        defaultOutputModes=["text", "text/plain"],
        capabilities=capabilities,
        skills=[skill],
    )

    server = A2AServer(
        agent_card=agent_card,
        task_manager=AgentTaskManager(agent=SentimentAnalysisAgent()),
        host=host,
        port=port,
    )
    logger.info(f"Starting server on {host}:{port}")
    server.start()
  except MissingAPIKeyError as e:
    logger.error(f"Error: {e}")
    exit(1)
  except Exception as e:
    logger.error(f"An error occurred during server startup: {e}")
    exit(1)


if __name__ == "__main__":
  main()
