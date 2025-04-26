"""Crew AI based sample for A2A protocol.

Handles the agents and also presents the tools required.
"""

import asyncio
import contextlib
from contextlib import AsyncExitStack
import logging
import os
import sys
import threading
from typing import Any, AsyncIterable, Dict, List, Optional, Sequence

# Explicitly import from the installed package by using the full path
# to avoid confusing with the local directory
import crewai
from crewai import Agent, Crew, Task 
from crewai.process import Process
from crewai.tools import tool
from crewai import LLM

from dotenv import load_dotenv
from google import genai
import json
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession
from mcp.types import TextContent
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class SentimentAnalysisResult(BaseModel):
  """Represents sentiment analysis results.

  Attributes:
    overall_sentiment: Overall sentiment score (-1 to 1 scale).
    positive_count: Number of positive posts/comments.
    negative_count: Number of negative posts/comments.
    neutral_count: Number of neutral posts/comments.
    summary: Text summary of the sentiment analysis.
    topics: Key topics discussed in the data.
  """
  overall_sentiment: float
  positive_count: int
  negative_count: int
  neutral_count: int
  summary: str
  topics: List[str]

def get_api_key() -> str:
  """Helper method to handle API Key."""
  load_dotenv()
  return os.getenv("GOOGLE_API_KEY")

async def _fetch_reddit_data_async(subreddit: str, limit: int = 15, session_id: str = None) -> str:
  """Fetch recent posts from a specified subreddit (async implementation using MCP).

  Args:
    subreddit: The name of the subreddit to fetch posts from (e.g., 'Bitcoin').
    limit: Maximum number of posts to fetch.
    session_id: Session identifier (for tracking purposes).

  Returns:
    JSON string containing post data or an error message.
  """
  if not subreddit:
    raise ValueError("Subreddit name cannot be empty")

  # Get the current directory of agent.py
  current_dir = os.path.dirname(os.path.abspath(__file__))
  # Navigate to the mcp-server-reddit directory
  server_path = os.path.normpath(os.path.join(current_dir, "..", "mcp-server-reddit"))
  server_script = os.path.join(server_path, "src", "mcp_server_reddit", "__main__.py")
  # Get the src directory of the server
  server_src_dir = os.path.join(server_path, "src") 

  exit_stack = AsyncExitStack()
  try:
    # Start the MCP server
    command = "python"

    # Create a modified environment to include virtual environment path
    env = os.environ.copy()
    # Use sys.prefix which points to the venv python is running from
    venv_path = sys.prefix
    # Construct the site-packages path based on OS
    if sys.platform == "win32":
        venv_site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
    else:
        # Use sys.version_info.major and sys.version_info.minor
        venv_site_packages = os.path.join(venv_path, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')

    # Add agent's venv site-packages AND server's src directory to PYTHONPATH
    python_paths = [venv_site_packages, server_src_dir]
    if 'PYTHONPATH' in env:
        # Append existing paths
        python_paths.append(env['PYTHONPATH'])
        
    env['PYTHONPATH'] = os.pathsep.join(python_paths)

    logger.info(f"Using PYTHONPATH: {env.get('PYTHONPATH')}")
    logger.info(f"MCP Server Script: {server_script}")
    logger.info(f"Using Python command: {command}")

    server_params = StdioServerParameters(
      command=command,
      args=[server_script],
      env=env,
      # Set the working directory to the server's src directory
      working_directory=server_src_dir
    )

    # Connect to the MCP server
    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))

    # Initialize the session
    await session.initialize()
    logger.info("MCP Session Initialized.")

    # Call the appropriate tool to fetch posts from the Bitcoin subreddit
    logger.info(f"Calling MCP tool 'get_subreddit_new_posts' for r/{subreddit} limit={limit}")
    result = await session.call_tool(
      "get_subreddit_new_posts",
      {
        "subreddit_name": subreddit,
        "limit": limit
      }
    )
    logger.info("MCP tool call completed.")

    # Convert the result to a readable format
    posts_data = []
    for item in result.content:
      if isinstance(item, TextContent):
        try:
            # Assuming item.text contains JSON for a single post/object
            post = json.loads(item.text)
            posts_data.append(post)
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON from MCP server: {item.text}. Error: {json_err}")
            # Include error info for the agent to potentially handle
            posts_data.append({"error": "Failed to decode MCP response", "raw_text": item.text})
      else:
          # Log unexpected content types
          logger.warning(f"Received unexpected content type from MCP server: {type(item)}")
          # Optionally include unexpected data if it can be represented as text/dict
          try:
              posts_data.append({"warning": "Unexpected content type", "data": str(item)})
          except Exception:
              posts_data.append({"warning": "Unexpected content type", "type": str(type(item))})


    logger.info(f"Fetched {len(posts_data)} items from r/{subreddit}")

    # Return the list of posts/items as a single JSON string
    return json.dumps(posts_data)
  except Exception as e:
    logger.exception(f"Error during async Reddit data fetch: {e}", exc_info=True)
    # Return a JSON string indicating the error
    return json.dumps({"error": f"Failed to fetch Reddit data: {str(e)}"})
  finally:
    logger.info("Closing MCP exit stack.")
    await exit_stack.aclose()
    logger.info("MCP exit stack closed.")

# Helper function to run async code in a separate thread
def run_async_in_thread(coro, args):
    result = None
    exception = None
    
    def target():
        nonlocal result, exception
        try:
            # Get the current event loop or create a new one
            try:
                # Try to get the current event loop 
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    # If it's closed, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                # If there's no event loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the coroutine in the event loop
            result = loop.run_until_complete(coro(*args))
            
            # Don't close the loop when done - it might be used elsewhere
        except Exception as e:
            exception = e
            
    thread = threading.Thread(target=target)
    thread.start()
    thread.join() # Wait for the thread to complete
    
    if exception:
        raise exception # Reraise exception in the main thread
    return result

@tool("RedditDataTool")
def fetch_reddit_data_tool(subreddit: str, limit: int = 15, session_id: str = None) -> str:
  """Fetch recent posts from a specified subreddit using MCP.

  Args:
    subreddit: The name of the subreddit to fetch posts from (e.g., 'Bitcoin').
    limit: Maximum number of posts to fetch.
    session_id: Session identifier (for tracking purposes).

  Returns:
    JSON string containing post data or an error message.
  """
  logger.info(f"Executing sync tool wrapper for RedditDataTool (r/{subreddit}, limit={limit})")
  try:
    # Run the async function in a separate thread using asyncio.run
    result_json = run_async_in_thread(_fetch_reddit_data_async, (subreddit, limit, session_id))
    logger.info(f"Sync wrapper received result (first 100 chars): {result_json[:100]}")
    return result_json
  except Exception as e:
      # Catch potential errors from the async function or thread execution
      logger.exception(f"Error running async tool fetch via thread wrapper: {e}", exc_info=True)
      # Return an error JSON string so the LLM knows something went wrong
      return json.dumps({"error": f"Failed to fetch Reddit data via wrapper: {str(e)}"})

class SentimentAnalysisAgent:
  """Agent that analyzes Bitcoin sentiment based on Reddit data."""

  def __init__(self):
    self.model = LLM(model="gemini/gemini-2.0-flash", api_key=get_api_key())

    self.sentiment_analyst_agent = Agent(
        role="Bitcoin Sentiment Analyst",
        goal=(
            "Analyze Reddit data to determine the current sentiment around Bitcoin. "
            "Provide a comprehensive analysis of how the Bitcoin community is feeling "
            "and what key topics are being discussed."
        ),
        backstory=(
            "You are a financial sentiment analyst specializing in cryptocurrency markets. "
            "You're skilled at analyzing social media content to gauge market sentiment "
            "and identify emerging trends in the Bitcoin ecosystem. Your analysis helps "
            "traders and investors understand the current mood of the community."
        ),
        verbose=False,
        allow_delegation=False,
        tools=[fetch_reddit_data_tool], # Use the sync wrapper tool
        llm=self.model,
    )

    self.sentiment_analysis_task = Task(
        description=(
            "Analyze recent Bitcoin discussions on Reddit to determine overall sentiment. "
            "Fetch the latest posts from the Bitcoin subreddit using the RedditDataTool. "
            "The tool will return a JSON string list of posts. Analyze the post titles, content, and metadata "
            "to determine whether each post expresses positive, negative, or neutral sentiment about Bitcoin. "
            "If the tool failed to retrieve data (indicated by an error message in the result), report the failure clearly. "
            "Otherwise, categorize posts by common themes or topics being discussed. "
            "Calculate an overall sentiment score and provide a summary of the current "
            "Bitcoin community sentiment. "
            "Identify key topics of discussion and any notable trends.\n\n"
            "IMPORTANT: DO NOT return the raw JSON data in your response. Instead, provide a "
            "comprehensive analysis in the following format:\n\n"
            "# Bitcoin Sentiment Analysis\n\n"
            "## Overall Sentiment: [Positive/Negative/Neutral]\n\n"
            "- **Sentiment Score**: [score] (on a scale from -1 to 1)\n"
            "- **Positive Posts**: [count]\n"
            "- **Negative Posts**: [count]\n"
            "- **Neutral Posts**: [count]\n"
            "- **Total Posts Analyzed**: [count]\n\n"
            "## Key Topics:\n"
            "[List of key topics]\n\n"
            "## Recent Discussion Topics:\n"
            "[List of interesting post titles]\n\n"
            "## Summary:\n"
            "[A paragraph summary of the overall sentiment and key findings]"
        ),
        expected_output=(
            "A comprehensive sentiment analysis of Bitcoin-related discussions on Reddit, "
            "including overall sentiment (positive, negative, or neutral), key topics, "
            "and a summary of the community\'s current outlook. If data fetching failed, the output should state the error encountered."
        ),
        agent=self.sentiment_analyst_agent,
    )

    self.sentiment_crew = Crew(
        agents=[self.sentiment_analyst_agent],
        tasks=[self.sentiment_analysis_task],
        process=Process.sequential,
        verbose=False, # Set to True for more detailed CrewAI logging if needed
    )

  def invoke(self, query, session_id) -> str:
    """Kickoff CrewAI and return the response."""

    # Default to Bitcoin subreddit if query doesn't specify otherwise
    subreddit = "Bitcoin"

    # Extract custom subreddit if specified in the query
    if "subreddit:" in query.lower():
      import re
      subreddit_match = re.search(r'subreddit:(\w+)', query, re.IGNORECASE)
      if subreddit_match:
        subreddit = subreddit_match.group(1)

    inputs = {
      "user_prompt": query,
      "session_id": session_id,
      "subreddit": subreddit # Pass subreddit to the task context if needed, although the tool takes it directly
    }

    logger.info(f"Invoking CrewAI with inputs: {inputs}")
    print(f"Starting sentiment analysis on r/{subreddit}")

    try:
      # CrewAI kickoff is synchronous and returns a CrewOutput object
      response = self.sentiment_crew.kickoff(inputs)
      
      # Access the .raw attribute for the string result before slicing for logging
      response_text = response.raw if response and hasattr(response, 'raw') else str(response)
      
      # Check if the response is raw JSON data
      if response_text.strip().startswith('[{') or response_text.strip().startswith('```\n[{'):
        logger.warning("Agent returned raw JSON instead of analysis. Returning error message.")
        return """
# Bitcoin Sentiment Analysis Error

The agent has returned raw data instead of performing sentiment analysis. This indicates an issue with the language model's response format.

Please try your query again or contact support if this issue persists.
"""
      
      logger.info(f"CrewAI kickoff completed. Response (first 100 chars): {response_text[:100]}")
      return response # Return the full CrewOutput object as task_manager expects
    except Exception as e:
      error_message = f"Error during sentiment analysis: {str(e)}"
      logger.exception(error_message)
      return error_message

  async def stream(self, query: str) -> AsyncIterable[Dict[str, Any]]:
    """Streaming is not supported by CrewAI."""
    raise NotImplementedError("Streaming is not supported by CrewAI.")
