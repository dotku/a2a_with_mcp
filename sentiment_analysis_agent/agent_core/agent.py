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
import sys
import importlib.util

# Check if crewai is installed in the system
if importlib.util.find_spec("crewai") is not None:
    # Import from the system crewai package
    from crewai.agent import Agent as CrewAIAgent
    from crewai.crew import Crew as CrewAICrew
    from crewai.task import Task as CrewAITask
    # Updated import - LLM is now in the root module
    from crewai import LLM as CrewAILLM
    from crewai.process import Process as CrewAIProcess
    # Import the tool decorator
    from crewai.tools import tool
else:
    print("Error: crewai package not found. Please install it with 'pip install crewai'")
    sys.exit(1)

from dotenv import load_dotenv
from google import genai
import json
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession
from mcp.types import TextContent
from pydantic import BaseModel

# Change to explicitly import from the installed crewai package
import crewai.agent
import crewai.crew
import crewai.task
# Use these as: crewai.agent.Agent, crewai.crew.Crew, etc.

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
  # Navigate to the mcp-server-reddit directory relative to the new module path
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
    limit: Maximum number of posts to fetch. Defaults to 50 if not specified by the agent.
    session_id: Session identifier (for tracking purposes).

  Returns:
    JSON string containing post data or an error message in JSON format: {"error": "description"}.
  """
  logger.info(f"Executing sync tool wrapper for RedditDataTool (r/{subreddit}, limit={limit})")
  # Add a default limit if the agent doesn't specify one, as the async function needs it.
  if limit is None or limit <= 0:
      limit = 50 # Set a reasonable default
      logger.info(f"Limit not specified or invalid, using default: {limit}")
      
  try:
    # Run the async function in a separate thread using asyncio.run
    result_json = run_async_in_thread(_fetch_reddit_data_async, (subreddit, limit, session_id))
    logger.info(f"Sync wrapper received result (first 100 chars): {result_json[:100]}")
    
    # Explicitly check for the specific MCP server error string
    if result_json == '[{"tool_error": ""}]':
        logger.warning("Detected specific tool_error from MCP server. Returning standard error JSON.")
        return json.dumps({"error": "The Reddit data tool failed to retrieve data from the underlying service."})
        
    # Also check if the result is already an error JSON from the async function
    try:
        data = json.loads(result_json)
        if isinstance(data, dict) and 'error' in data:
            logger.warning(f"Returning existing error JSON from async fetch: {data['error']}")
            return result_json # Pass the existing error JSON through
    except json.JSONDecodeError:
        # If it's not valid JSON and not the specific tool_error string, something else is wrong
        logger.error(f"Received non-JSON, non-error response from async tool: {result_json[:200]}")
        return json.dumps({"error": "Received unexpected or invalid data format from the Reddit data tool."})
        
    # If it looks like valid data, return it
    return result_json
  except Exception as e:
      # Catch potential errors from the async function or thread execution
      logger.exception(f"Error running async tool fetch via thread wrapper: {e}", exc_info=True)
      # Return an error JSON string so the LLM knows something went wrong
      return json.dumps({"error": f"Failed to fetch Reddit data via wrapper: {str(e)}"})

class SentimentAnalysisAgent:
  """Agent that analyzes Bitcoin sentiment based on Reddit data."""

  def __init__(self):
    self.model = CrewAILLM(model="gemini/gemini-2.0-flash-lite", api_key=get_api_key())
    
    # --- Define Templates for dynamic roles/goals/tasks ---
    self.agent_role_template = "{crypto_name} Sentiment Analyst"
    self.agent_goal_template = (
        "Analyze Reddit data from r/{subreddit} to determine the current sentiment around {crypto_name}. "
        "Provide a comprehensive analysis of how the {crypto_name} community is feeling "
        "and what key topics are being discussed."
    )
    self.agent_backstory_template = (
        "You are a financial sentiment analyst specializing in cryptocurrency markets. "
        "You're skilled at analyzing social media content from subreddits like r/{subreddit} to gauge market sentiment "
        "and identify emerging trends in the {crypto_name} ecosystem. Your analysis helps "
        "traders and investors understand the current mood of the community."
    )
    self.task_description_template = (
        "Analyze recent {crypto_name} discussions on Reddit (specifically from r/{subreddit}) to determine overall sentiment. "
        "Fetch the latest posts from the {subreddit} subreddit using the RedditDataTool. "
        "The tool will return a JSON string list of posts. Analyze the post titles, content, and metadata "
        "to determine whether each post expresses positive, negative, or neutral sentiment about {crypto_name}. "
        "If the tool failed to retrieve data (indicated by an error message in the result), report the failure clearly. "
        "Otherwise, categorize posts by common themes or topics being discussed. "
        "Calculate an overall sentiment score and provide a summary of the current "
        "{crypto_name} community sentiment. "
        "Identify key topics of discussion and any notable trends.\n\n"
        "IMPORTANT: DO NOT return the raw JSON data in your response. Instead, provide a "
        "comprehensive analysis in the following format:\n\n"
        "# {crypto_name} Sentiment Analysis (from r/{subreddit})\n\n"
        "## Overall Sentiment: [Positive/Negative/Neutral]\n\n"
        "- **Sentiment Score**: [score] (on a scale from -1 to 1)\n"
        "- **Positive Posts**: [count]\n"
        "- **Negative Posts**: [count]\n"
        "- **Neutral Posts**: [count]\n"
        "- **Total Posts Analyzed**: [count]\n\n"
        "## Key Topics:\n"
        "[List of key topics relevant to {crypto_name}]\n\n"
        "## Recent Discussion Topics:\n"
        "[List of interesting post titles from r/{subreddit}]\n\n"
        "## Summary:\n"
        "[A paragraph summary of the overall sentiment and key findings for {crypto_name} based on r/{subreddit}]"
    )
    self.task_expected_output_template = (
        "A comprehensive sentiment analysis of {crypto_name}-related discussions on Reddit (from r/{subreddit}), "
        "including overall sentiment (positive, negative, or neutral), key topics, "
        "and a summary of the community's current outlook. If data fetching failed, the output should state the error encountered."
    )
    # ----------------------------------------------------------

  def invoke(self, query, session_id) -> str:
    """Kickoff CrewAI and return the response."""
    
    # --- Parse Crypto and Subreddit from query --- 
    query_lower = query.lower().strip()
    crypto_name = "Bitcoin" # Default
    subreddit = "Bitcoin" # Default
    import re

    # Define known crypto names/tickers and their canonical names/subreddits
    crypto_map = {
        "bitcoin": ("Bitcoin", "Bitcoin"),
        "btc": ("Bitcoin", "Bitcoin"),
        "ethereum": ("Ethereum", "ethereum"),
        "eth": ("Ethereum", "ethereum"),
        "ripple": ("XRP", "XRP"),
        "xrp": ("XRP", "XRP"),
        "solana": ("Solana", "solana"),
        "sol": ("Solana", "solana"),
        "dogecoin": ("Dogecoin", "dogecoin"),
        "doge": ("Dogecoin", "dogecoin"),
    }

    found_crypto = False
    # Iterate through known cryptos (longer names first for better matching)
    sorted_keys = sorted(crypto_map.keys(), key=len, reverse=True)
    for key in sorted_keys:
        # Use word boundaries to avoid partial matches (e.g., 'eth' in 'something')
        # Check for space padding, start, end, or exact match
        if f" {key} " in f" {query_lower} " or query_lower.startswith(f"{key} ") or query_lower.endswith(f" {key}") or query_lower == key:
            c_name, s_name = crypto_map[key]
            crypto_name = c_name
            subreddit = s_name # Default subreddit for this crypto
            logger.info(f"Found crypto '{key}', setting defaults: Crypto={crypto_name}, Subreddit={subreddit}")
            found_crypto = True
            break # Stop after first match

    # Now, try to find an *explicit* subreddit mention to override the default
    subreddit_match = re.search(r"(?:subreddit|r/)\s*([\w.-]+)", query_lower)
    if subreddit_match:
        parsed_subreddit = subreddit_match.group(1)
        # Use the explicitly found subreddit name
        # Optional: Could validate against known subreddits if needed
        subreddit = parsed_subreddit
        logger.info(f"Found explicit subreddit mention, overriding default: Subreddit='{subreddit}'")
    elif not found_crypto:
         logger.warning(f"Could not identify crypto or explicit subreddit in query: '{query}'. Falling back to Bitcoin defaults.")
         # Ensure defaults are explicitly set if no crypto was found
         crypto_name = "Bitcoin"
         subreddit = "Bitcoin"
    else:
         logger.info(f"No explicit subreddit found, using default '{subreddit}' for crypto '{crypto_name}'.")

    logger.info(f"Final Parsed Values: Crypto={crypto_name}, Subreddit={subreddit}")
    # ---------------------------------------------
    
    # --- Dynamically create Agent and Task --- 
    dynamic_role = self.agent_role_template.format(crypto_name=crypto_name)
    dynamic_goal = self.agent_goal_template.format(crypto_name=crypto_name, subreddit=subreddit)
    dynamic_backstory = self.agent_backstory_template.format(crypto_name=crypto_name, subreddit=subreddit)
    dynamic_description = self.task_description_template.format(crypto_name=crypto_name, subreddit=subreddit)
    dynamic_expected_output = self.task_expected_output_template.format(crypto_name=crypto_name, subreddit=subreddit)
    
    sentiment_analyst_agent = CrewAIAgent(
        role=dynamic_role,
        goal=dynamic_goal,
        backstory=dynamic_backstory,
        verbose=False,
        allow_delegation=False,
        tools=[fetch_reddit_data_tool], # Use the sync wrapper tool
        llm=self.model,
    )

    sentiment_analysis_task = CrewAITask(
        description=dynamic_description,
        expected_output=dynamic_expected_output,
        agent=sentiment_analyst_agent,
        # Remove context parameter - it expects a List[Task] and isn't needed here
        # The subreddit is passed via the 'inputs' dict below and used by the tool directly.
        # context={"subreddit": subreddit} 
    )

    sentiment_crew = CrewAICrew(
        agents=[sentiment_analyst_agent],
        tasks=[sentiment_analysis_task],
        process=CrewAIProcess.sequential,
        verbose=False, # Set to True for more detailed CrewAI logging if needed
    )
    # -----------------------------------------

    # Use parsed values in kickoff inputs
    inputs = {
      "user_prompt": query, # Original query for context
      "session_id": session_id,
      "crypto_name": crypto_name,
      "subreddit": subreddit
    }

    logger.info(f"Invoking CrewAI with inputs: {inputs}")
    print(f"Starting sentiment analysis on r/{subreddit} for {crypto_name}")

    try:
      # Use the dynamically created crew
      response = sentiment_crew.kickoff(inputs)
      
      # Access the .raw attribute for the string result
      response_text = response.raw if response and hasattr(response, 'raw') else str(response)
      logger.info(f"CrewAI kickoff completed. Raw response (first 100 chars): {response_text[:100]}")
      
      # --- Improved Error Handling --- 
      # Check if the raw response indicates a tool error explicitly
      # The tool now returns JSON {"error": ...} on failure
      tool_error_detected = False
      error_message = "An unknown error occurred during sentiment analysis."
      try:
          # Attempt to parse the raw response as JSON
          parsed_response = json.loads(response_text)
          if isinstance(parsed_response, dict) and 'error' in parsed_response:
              tool_error_detected = True
              error_message = f"Error from data tool: {parsed_response['error']}"
              logger.error(f"Detected error JSON in CrewAI response: {error_message}")
          # Check if the response is the unwanted raw JSON input data that was previously returned
          elif isinstance(parsed_response, dict) and all(k in parsed_response for k in ['subreddit', 'limit', 'session_id']):
               tool_error_detected = True # Treat this as an error (agent didn't format output) 
               error_message = "Agent returned raw input data instead of analysis. Possible tool failure or LLM formatting issue."
               logger.error(error_message)
               
      except json.JSONDecodeError:
          # If it's not JSON, it might be the formatted analysis or a simple text error message
          # Check for common error phrases just in case
          if "error" in response_text.lower() or "failed" in response_text.lower():
              # Heuristic check - might not always be accurate
              # Consider if the LLM itself generated an error message in text format
              # For now, we assume if it contains 'error' or 'failed', something went wrong.
              # Let's prioritize the JSON check above.
              pass # Handled by the JSON check or will be returned as is if it's intended text
          pass 
      except Exception as parse_exc:
          # Catch any other parsing issues
          logger.error(f"Unexpected error parsing CrewAI response: {parse_exc}")
          tool_error_detected = True
          error_message = f"Internal error processing agent response: {parse_exc}"

      if tool_error_detected:
            # Return a structured error message if the tool failed or format was wrong
            return json.dumps({
                "status": "error",
                "message": error_message
            })
            
      # Check if the response looks like the analysis was skipped (e.g., just a code block)
      # This is a heuristic check based on the previous bad output observed
      if response_text.strip().startswith("```tool_code") or response_text.strip().startswith("```json") :
            logger.warning("Agent returned raw code/JSON block instead of analysis. Returning error message.")
            return json.dumps({
                "status": "error",
                "message": "Agent returned raw data/code instead of performing sentiment analysis. This may indicate an issue with the underlying tool or response formatting."
            })
            
      # If no errors detected and not the raw code block, return the presumed successful analysis
      # The task manager expects the raw string here, it will wrap it in the artifact. 
      return response_text 
      
    except Exception as e:
      error_message = f"Error during sentiment analysis execution: {str(e)}"
      logger.exception(error_message, exc_info=True)
      # Return a structured error
      return json.dumps({
          "status": "error",
          "message": error_message
      })

  async def stream(self, query: str) -> AsyncIterable[Dict[str, Any]]:
    """Streaming is not supported by CrewAI."""
    raise NotImplementedError("Streaming is not supported by CrewAI.")
