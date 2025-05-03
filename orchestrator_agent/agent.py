import json
import uuid
from typing import Any, AsyncIterable, Dict, List, Optional
import traceback
import logging
import time
import httpx
import requests

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Configure logging
logger = logging.getLogger(__name__)

# Process request function for A2A
def process_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Process a request from the A2A server."""
    try:
        logger.info(f"Processing orchestrator request: {request}")
        
        # Create orchestrator agent if needed
        orchestrator = OrchestratorAgent()
        
        # Extract query and session ID
        query = request.get("query", "")
        session_id = request.get("session_id", str(uuid.uuid4()))
        
        # Process the request using the orchestrator agent
        result = orchestrator.invoke(query, session_id)
        
        # Return the result
        return {
            "status": "completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e)
        }

class OrchestratorAgent:
    """
    Orchestrator Agent that breaks down user tasks and delegates them to specialized agents.
    """

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        try:
            self._agent = self._build_agent()
            self._user_id = "remote_agent"
            self._runner = Runner(
                app_name=self._agent.name,
                agent=self._agent,
                artifact_service=InMemoryArtifactService(),
                session_service=InMemorySessionService(),
                memory_service=InMemoryMemoryService(),
            )
            # URLs for specialized agents - Update financial_data URL
            self.agent_urls = {
                "financial_data": "http://localhost:8001", # Updated port
                "sentiment_analysis": "http://localhost:10000",
                "competitor_analysis": "http://localhost:8003",
                "visualization": "http://localhost:8004",
                "prompt_templates": "http://localhost:8005"
            }
            logger.info("Orchestrator agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing orchestrator agent: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def invoke(self, query, session_id) -> str:
        """
        Invoke the agent synchronously.
        """
        try:
            session = self._runner.session_service.get_session(
                app_name=self._agent.name, user_id=self._user_id, session_id=session_id
            )
            content = types.Content(
                role="user", parts=[types.Part.from_text(text=query)]
            )
            if session is None:
                session = self._runner.session_service.create_session(
                    app_name=self._agent.name,
                    user_id=self._user_id,
                    state={},
                    session_id=session_id,
                )
            events = list(self._runner.run(
                user_id=self._user_id, session_id=session.id, new_message=content
            ))
            if not events or not events[-1].content or not events[-1].content.parts:
                return ""
            return "\n".join([p.text for p in events[-1].content.parts if hasattr(p, 'text') and p.text])
        except Exception as e:
            logger.error(f"Error in invoke: {str(e)}")
            logger.error(traceback.format_exc())
            return json.dumps({"error": str(e)})

    async def stream(self, query, session_id) -> AsyncIterable[Dict[str, Any]]:
        """
        Invoke the agent with streaming responses.
        """
        try:
            # Yield an initial update to ensure the stream starts immediately
            yield {
                "is_task_complete": False,
                "updates": "Starting orchestration process..."
            }
            
            # Get or create session
            session = self._runner.session_service.get_session(
                app_name=self._agent.name, user_id=self._user_id, session_id=session_id
            )
            
            # Create query content
            content = types.Content(
                role="user", parts=[types.Part.from_text(text=query)]
            )
            
            # Create session if it doesn't exist
            if session is None:
                session = self._runner.session_service.create_session(
                    app_name=self._agent.name,
                    user_id=self._user_id,
                    state={},
                    session_id=session_id,
                )
                
            # Log the session creation
            logger.info(f"Session created for task {session_id}")
            
            # Yield progress update
            yield {
                "is_task_complete": False,
                "updates": "Orchestrator is analyzing your request..."
            }
            
            # Track if we've had any updates
            update_count = 0
            
            # Run the agent asynchronously
            try:
                async for event in self._runner.run_async(
                    user_id=self._user_id, session_id=session.id, new_message=content
                ):
                    update_count += 1
                    logger.info(f"Received event for task {session_id}: {type(event)}")
                    
                    if event.is_final_response():
                        response = ""
                        if (
                            event.content
                            and event.content.parts
                            and any([p.text for p in event.content.parts if hasattr(p, 'text') and p.text])
                        ):
                            response = "\n".join([p.text for p in event.content.parts if hasattr(p, 'text') and p.text])
                        elif (
                            event.content
                            and event.content.parts
                            and any([True for p in event.content.parts if hasattr(p, 'function_response') and p.function_response])
                        ):
                            response = next((p.function_response.model_dump() for p in event.content.parts 
                                            if hasattr(p, 'function_response') and p.function_response), {})
                        
                        logger.info(f"Yielding final response for task {session_id}")
                        yield {
                            "is_task_complete": True,
                            "content": response,
                        }
                    else:
                        message = None
                        if hasattr(event, 'input') and event.input and event.input.parts:
                            for part in event.input.parts:
                                if hasattr(part, 'text') and part.text:
                                    message = part.text
                                    break
                        
                        status_message = f"Orchestrating your request: {message}" if message else "Processing your request..."
                        
                        logger.info(f"Yielding update for task {session_id}: {status_message}")
                        yield {
                            "is_task_complete": False,
                            "updates": status_message,
                        }
            except Exception as inner_e:
                logger.error(f"Error in run_async: {str(inner_e)}")
                logger.error(traceback.format_exc())
                yield {
                    "is_task_complete": True,
                    "content": json.dumps({
                        "error": f"Error in agent execution: {str(inner_e)}",
                        "status": "error"
                    }),
                }
            
            # If no updates were generated, yield a final update 
            if update_count == 0:
                logger.info(f"No updates generated for task {session_id}, sending fallback response")
                yield {
                    "is_task_complete": True,
                    "content": json.dumps({
                        "message": "Task completed but no specific updates were generated.",
                        "status": "completed"
                    }),
                }
                
        except Exception as e:
            logger.error(f"Error in stream: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Yield error as a complete response
            yield {
                "is_task_complete": True,
                "content": json.dumps({
                    "error": str(e),
                    "status": "error"
                }),
            }

    def _build_agent(self) -> LlmAgent:
        """Builds the LLM agent for the orchestrator."""
        try:
            return LlmAgent(
                model="gemini-2.0-flash-lite",
                name="orchestrator_agent",
                description=(
                    "This agent breaks down complex tasks into smaller subtasks and delegates them "
                    "to specialized agents. It then aggregates the results and provides a final response."
                ),
                # Updated instruction to detail all financial agent tools AND sentiment mapping
                instruction=(
                    "You are an orchestrator agent that helps perform market research and investment analysis.\n\n"
                    "When you receive a request, break it down into specific subtasks and delegate them \n"
                    "to the appropriate specialized agents:\n\n"
                    "1. Financial Data Agent: For retrieving financial statements, stock price history, running SQL queries on financial data, AND performing cryptocurrency market and historical analysis.\n"
                    "2. Sentiment Analysis Agent: For analyzing news and social media sentiment for specific cryptocurrencies. Use the mapping below for subreddits:\n"
                    "    - Bitcoin (BTC): r/Bitcoin\n"
                    "    - Ethereum (ETH): r/ethereum\n"
                    "    - Ripple (XRP): r/XRP\n"
                    "    - Solana (SOL): r/solana\n"
                    "    - Dogecoin (DOGE): r/dogecoin\n"
                    "3. Competitor Analysis Agent: For analyzing competitor data and market positioning\n"
                    "4. Visualization Agent: For creating visual representations of data\n"
                    "5. Prompt Templates Agent: For providing standardized analysis templates\n\n"
                    "After collecting information from these specialized agents, synthesize the results \n"
                    "into a comprehensive response for the user.\n\n"
                    "VISUALIZATION HANDLING - MOST IMPORTANT: When the user asks for a visualization, you should:\n\n"
                    "1. Call the generate_visualization tool\n"
                    "2. When you receive the result, DO NOT say you can't 'display', 'show', or 'provide' the visualization\n"
                    "3. INSTEAD, respond with something like: 'I've created a visualization of [describe what the visualization shows]. The chart displays [key insights from data]. Here are the details:'\n"
                    "4. Then list the key data points in a structured format (e.g., for a bar chart, list each category and its value)\n"
                    "5. GOOD RESPONSE EXAMPLE: 'I've created a bar chart showing Number of Users for different Service Names. College Finder has the most users at 850, followed by Scholarship Hub with 670 users...'\n"
                    "6. BAD RESPONSE EXAMPLE: 'I'm unable to display the chart, but it shows the number of users...'\n"
                    "7. Remember, describing the visualization IS providing it to the user. This is exactly what is expected.\n\n"
                    "You can use the following tools to interact with specialized agents:\n"
                    "- fetch_financial_statements: Delegate task to Financial Agent: Request financial statements (income, balance sheet, cash flow) for a company and year.\n"
                    "- fetch_stock_price_history: Delegate task to Financial Agent: Request historical stock prices for a ticker symbol within an optional date range. Omit dates for current price.\n"
                    "- run_sql_query: Delegate task to Financial Agent: Execute a read-only SQL query (SELECT or WITH) on the financial database.\n"
                    "- get_crypto_market_analysis: Delegate task to Financial Agent: Request detailed market analysis for a cryptocurrency (top exchanges, volume, VWAP).\n"
                    "- get_crypto_historical_analysis: Delegate task to Financial Agent: Request historical price analysis for a cryptocurrency with optional timeframe.\n"
                    "- fetch_news_sentiment: Delegate task to Sentiment Analysis Agent: Analyze news/social media sentiment for a company/crypto and subreddit. Determine the correct subreddit using the mapping above. If the user asks for the 'latest' sentiment, use the default timeframe ('latest').\n"
                    "- analyze_competitors: Delegate task to Competitor Analysis Agent: Gather competitor information.\n"
                    "- generate_visualization: Delegate task to Visualization Agent: Create visual representations of data. Your response should describe what the visualization shows in detail.\n"
                    "- get_analysis_template: Delegate task to Prompt Templates Agent: Obtain standardized analysis templates.\n"
                ),
                # Updated tools list with new crypto analysis tools
                tools=[
                    self.fetch_financial_statements,
                    self.fetch_stock_price_history,
                    self.run_sql_query,
                    self.get_crypto_market_analysis,
                    self.get_crypto_historical_analysis,
                    self.fetch_news_sentiment,
                    self.analyze_competitors,
                    self.generate_visualization,
                    self.get_analysis_template,
                ],
            )
        except Exception as e:
            logger.error(f"Error building LLM agent: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    # Helper method for making A2A calls with polling
    def _call_specialized_agent(self, agent_name=None, agent_url=None, agent_input=None, query_text=None, tool_context=None):
        """Call a specialized agent using JSON-RPC."""
        if not agent_name and not agent_url:
            raise ValueError("Either agent_name or agent_url must be provided")
            
        if not agent_url:
            agent_url = self.agent_urls.get(agent_name, f"http://localhost:8000")
            
        # Handle different input cases
        final_input = None
        
        # If ToolContext is provided, extract relevant information
        if tool_context:
            logger.warning("ToolContext was passed directly to _call_specialized_agent. This may cause serialization issues. Consider using structured input instead.")
            if hasattr(tool_context, 'request') and hasattr(tool_context.request, 'body'):
                # Try to extract the original request body if available
                try:
                    final_input = json.dumps({"query": str(tool_context.request.body)})
                except:
                    final_input = json.dumps({"query": "Request from Orchestrator"})
        
        # Structured input dictionary (preferred)
        elif agent_input:
            # If agent_input is already a string, use it directly
            if isinstance(agent_input, str):
                final_input = agent_input
            else:
                # Convert dictionary to JSON string
                final_input = json.dumps(agent_input)
                
        # Otherwise use query_text as a simple string wrapped in a query field
        elif query_text:
            final_input = json.dumps({"query": query_text})
        else:
            final_input = json.dumps({"query": "Request from Orchestrator"})
            
        logger.info(f"Calling specialized agent with input: {final_input[:100]}...")
        
        try:
            # Generate a unique task ID using a UUID
            task_id = f"task-{str(uuid.uuid4())}"
            session_id = str(uuid.uuid4())
            
            # Prepare the JSON-RPC message
            message = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tasks/send",
                "params": {
                    "id": task_id,
                    "sessionId": session_id,
                    "acceptedOutputModes": ["text"],
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": final_input}]
                    }
                }
            }
            
            logger.info(f"Sending task/send request to Specialized Agent: {json.dumps(message)}")
            
            # Send the request
            response = requests.post(agent_url, json=message)
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            logger.info(f"Received task/send response from Specialized Agent: {json.dumps(response_data)}")
            
            # Check if the task was created successfully
            if "result" in response_data and "id" in response_data["result"]:
                task_info = response_data["result"]
                
                # Check the task status
                if task_info.get("status", {}).get("state") == "completed":
                    logger.info(f"Task {task_id} reported as COMPLETED in initial response. Extracting result.")
                    
                    # Extract the result from artifacts
                    artifacts = task_info.get("artifacts", [])
                    if artifacts:
                        # Get the first artifact's text content
                        for artifact in artifacts:
                            parts = artifact.get("parts", [])
                            for part in parts:
                                if part.get("type") == "text":
                                    result_text = part.get("text", "")
                                    logger.info(f"Extracted result from artifact for task {task_id}: {result_text[:100]}...")
                                    
                                    # Process the result to handle errors and special formats
                                    try:
                                        # If "Unable to retrieve sentiment data" is in the response, 
                                        # this is a user-friendly error message from sentiment analysis
                                        if "Unable to retrieve sentiment data" in result_text:
                                            logger.warning("Sentiment analysis failed with a user-friendly error message")
                                            # This is already in a good format for end users, wrap it in a success JSON
                                            return json.dumps({"status": "success", "data": result_text})
                                            
                                        # Try to parse the result as JSON (to detect error responses)
                                        parsed_result = json.loads(result_text)
                                        
                                        # If it's a success/data structure, return it as is
                                        if "status" in parsed_result and parsed_result["status"] == "success":
                                            return result_text
                                            
                                        # If it contains an explicit error message
                                        if "error" in parsed_result or "status" in parsed_result and parsed_result["status"] == "error":
                                            error_msg = parsed_result.get("error") or parsed_result.get("message", "Unknown error")
                                            logger.error(f"Error received from specialized agent: {error_msg}")
                                            # Format as a standardized error response
                                            return json.dumps({"status": "error", "message": f"Error from {agent_name} agent: {error_msg}"})
                                        
                                        # If we get here, it's valid JSON but not in our standard format
                                        # Return it wrapped in a success structure
                                        return json.dumps({"status": "success", "data": result_text})
                                        
                                    except json.JSONDecodeError:
                                        # Not JSON, just plain text result
                                        # Otherwise, attempt to parse JSON
                                        try:
                                            parsed_data = json.loads(result_text)
                                            # Return the already-JSON string if it parsed successfully
                                            return result_text
                                        except json.JSONDecodeError:
                                            # Not JSON, return as text data wrapped in a success JSON structure
                                            return json.dumps({"status": "success", "data": result_text})
                    # Fallback: Extract result from status message if no artifacts (less common)
                    elif ("status" in task_info
                            and isinstance(task_info.get("status"), dict) 
                            and "message" in task_info["status"]
                            and isinstance(task_info["status"].get("message"), dict)
                            and "parts" in task_info["status"]["message"]
                            and isinstance(task_info["status"]["message"].get("parts"), list)
                            and task_info["status"]["message"]["parts"]):
                        part = task_info["status"]["message"]["parts"][0]
                        if part.get("type") == "text":
                            result_text = part.get("text", "")
                            logger.info(f"Extracted result from status message for task {task_id}: {result_text[:100]}...")
                            return json.dumps({"status": "success", "data": result_text})
                            
            # If no valid result was found
            error_msg = f"No valid result found in specialized agent response"
            logger.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
            
        except requests.RequestException as e:
            error_msg = f"Request error calling {agent_url}: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})
            
        except Exception as e:
            error_msg = f"Error calling specialized agent: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"status": "error", "message": error_msg})

    # --- Helper methods to extract data from task results ---
    def _extract_result_from_task_data(self, task_data: Dict, task_id: str) -> str:
        """Extracts the primary result text from completed task data."""
        # First check for image artifacts (for visualization agent)
        if (
            "artifacts" in task_data 
            and isinstance(task_data.get("artifacts"), list) 
            and task_data["artifacts"] # not empty
        ):
            artifact = task_data["artifacts"][0]
            if isinstance(artifact, dict) and "parts" in artifact and artifact["parts"]:
                part = artifact["parts"][0]
                if isinstance(part, dict) and "file" in part:
                    file_data = part["file"]
                    if isinstance(file_data, dict) and "mimeType" in file_data and "bytes" in file_data:
                        # This is an image file - return it directly
                        logger.info(f"Found image file in response for task {task_id}")
                        result_json = {
                            "status": "success",
                            "image_data": {
                                "name": file_data.get("name", "plot.png"),
                                "mime_type": file_data.get("mimeType", "image/png"),
                                "data": file_data.get("bytes", "")
                            },
                            "message": "Visualization generated successfully. The image is available as base64-encoded data."
                        }
                        return json.dumps(result_json)
        # Fallback to text artifacts
        if (
            "artifacts" in task_data 
            and isinstance(task_data.get("artifacts"), list) 
            and task_data["artifacts"] # not empty
            and isinstance(task_data["artifacts"][0], dict)
                                     and "parts" in task_data["artifacts"][0]
                                     and isinstance(task_data["artifacts"][0].get("parts"), list)
                                     and task_data["artifacts"][0]["parts"] # not empty
                                     and isinstance(task_data["artifacts"][0]["parts"][0], dict)
                                     and "text" in task_data["artifacts"][0]["parts"][0]
                                ):
                                    result_text = task_data["artifacts"][0]["parts"][0]["text"]
                                    logger.info(f"Extracted result from artifact for task {task_id}: {result_text[:100]}...")
                                    # If this is from visualization agent, don't parse as JSON
                                    if "visualization" in result_text.lower() or "plot" in result_text.lower() or "chart" in result_text.lower():
                                        return json.dumps({
                                            "status": "success",
                                            "data": result_text,
                                            "message": "Visualization task completed successfully"
                                        })
                                    # Otherwise, attempt to parse JSON
                                    try:
                                        parsed_data = json.loads(result_text)
                                        # Return the already-JSON string if it parsed successfully
                                        return result_text
                                    except json.JSONDecodeError:
                                        # Not JSON, return as text data wrapped in a success JSON structure
                                        return json.dumps({"status": "success", "data": result_text})
        # Fallback: Extract result from status message if no artifacts (less common)
        elif (
            "status" in task_data
            and isinstance(task_data.get("status"), dict) 
            and "message" in task_data["status"]
            and isinstance(task_data["status"].get("message"), dict)
            and "parts" in task_data["status"]["message"]
            and isinstance(task_data["status"]["message"].get("parts"), list)
            and task_data["status"]["message"]["parts"] # not empty
            and isinstance(task_data["status"]["message"]["parts"][0], dict)
            and "text" in task_data["status"]["message"]["parts"][0]
        ):
            result_text = task_data["status"]["message"]["parts"][0]["text"]
            logger.info(f"Extracted result from status message for task {task_id}: {result_text[:100]}...")
            
            # Special handling for visualization responses
            if "plot generated" in result_text.lower() or "visualization" in result_text.lower():
                return json.dumps({
                    "status": "success", 
                    "message": result_text,
                    "type": "visualization"
                })
            
            # Default handling
            return json.dumps({"status": "success", "data": result_text})
        else:
            logger.warning(f"Task {task_id} completed but no result found in artifacts or status message.")
            return json.dumps({"status": "completed_no_data", "message": "Task completed but no standard result data found."})

    def _extract_error_from_task_data(self, task_data: Dict) -> str:
        """Extracts the error message from failed task data."""
        error_message = "Task failed for unknown reason."
        if (
            "status" in task_data
            and isinstance(task_data.get("status"), dict)
            and "message" in task_data["status"]
            and isinstance(task_data["status"].get("message"), dict)
            and "parts" in task_data["status"]["message"]
            and isinstance(task_data["status"]["message"].get("parts"), list)
            and task_data["status"]["message"]["parts"] # not empty
            and isinstance(task_data["status"]["message"]["parts"][0], dict)
            and "text" in task_data["status"]["message"]["parts"][0]
        ):
            error_message = task_data["status"]["message"]["parts"][0]["text"]
        return error_message
    # --- End Helper methods ---

    # Tool implementations
    # Renamed from fetch_financial_data
    def fetch_financial_statements(self, company: str, fiscal_year: int, tool_context: ToolContext) -> str:
        """
        Fetch financial statements (income statement, balance sheet, cash flow) for a given company and fiscal year using the Financial Agent.

        Args:
            company: The company ticker symbol or identifier
            fiscal_year: The fiscal year for which to fetch data

        Returns:
            Financial statement data for the specified company and fiscal year as a JSON string.
        """
        query_text = f"Fetch financial statements for {company} in {fiscal_year}"
        
        # Create a structured dictionary for the agent input
        financial_statements_request = {
            "query": query_text,
            "company": company,
            "fiscal_year": fiscal_year,
            "action": "fetch_financial_statements"
        }
        
        # Call the specialized agent with the structured request
        return self._call_specialized_agent(
            agent_name="financial_data", 
            agent_input=financial_statements_request,
            query_text=query_text,
            tool_context=None  # Don't pass the tool_context directly
        )

    # Added tool for stock price history
    def fetch_stock_price_history(self, ticker: str, tool_context: ToolContext, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        """
        Fetch historical stock prices for a given ticker symbol and date range using the Financial Agent.
        If start_date and end_date are omitted, it fetches the current stock information.

        Args:
            ticker: The stock ticker symbol (e.g., 'AAPL', 'GOOG')
            tool_context: The context provided by the ADK framework.
            start_date: Optional. The start date for the history (e.g., '2023-01-01'). Omit for current price.
            end_date: Optional. The end date for the history (e.g., '2023-12-31'). Omit for current price.

        Returns:
            Historical stock price data OR current stock info as a JSON string.
        """
        # Determine if it's a current price request based on missing dates
        if start_date is None or end_date is None:
            logger.info(f"Detected current price request for {ticker} due to missing dates.")
            query_text = f"Get current stock info for {ticker}"
            action = "get_current_stock_info"
        else:
            # Proceed with historical data request
            query_text = f"Fetch stock price history for {ticker} from {start_date} to {end_date}"
            action = "fetch_stock_price_history"
            
        # Create a structured dictionary for the agent input
        stock_price_request = {
            "query": query_text,
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "action": action
        }
        
        # Call the specialized agent with the structured request
        return self._call_specialized_agent(
            agent_name="financial_data", 
            agent_input=stock_price_request,
            query_text=query_text,
            tool_context=None  # Don't pass the tool_context directly
        )

    # Added tool for SQL query
    def run_sql_query(self, query: str, tool_context: ToolContext) -> str:
        """
        Execute a read-only SQL query on the financial database using the Financial Agent. Only SELECT or WITH queries are allowed.

        Args:
            query: The SQL query string to execute.

        Returns:
            The result of the SQL query as a JSON string.
        """
        # The query text sent to the financial agent should clearly indicate the intent to run SQL
        query_text = f"Run SQL query: {query}"
        
        # Create a structured dictionary for the agent input
        sql_query_request = {
            "query": query_text,
            "sql": query,
            "action": "run_sql_query"
        }
        
        # Call the specialized agent with the structured request
        return self._call_specialized_agent(
            agent_name="financial_data", 
            agent_input=sql_query_request,
            query_text=query_text,
            tool_context=None  # Don't pass the tool_context directly
        )

    # --- NEW: Tool for Crypto Market Analysis ---
    def get_crypto_market_analysis(self, symbol: str, tool_context: ToolContext) -> str:
        """
        Request detailed market analysis for a specific cryptocurrency using the Financial Agent.

        Args:
            symbol: The cryptocurrency symbol (e.g., 'BTC', 'ETH')
            tool_context: The context provided by the ADK framework.

        Returns:
            Market analysis data as a JSON string.
        """
        query_text = f"Perform market analysis for {symbol}"
        logger.info(f"Delegating market analysis for {symbol} to financial agent.")
        
        # Create a structured dictionary for the agent input
        market_analysis_request = {
            "query": query_text,
            "symbol": symbol,
            "action": "market_analysis"
        }
        
        # Call the specialized agent with the structured request
        return self._call_specialized_agent(
            agent_name="financial_data", 
            agent_input=market_analysis_request,
            query_text=query_text,
            tool_context=None  # Don't pass the tool_context directly
        )

    # --- NEW: Tool for Crypto Historical Analysis ---
    def get_crypto_historical_analysis(self, symbol: str, tool_context: ToolContext, interval: Optional[str] = None, days: Optional[int] = None) -> str:
        """
        Request historical price analysis for a specific cryptocurrency with optional timeframe using the Financial Agent.

        Args:
            symbol: The cryptocurrency symbol (e.g., 'BTC', 'ETH').
            tool_context: The context provided by the ADK framework.
            interval: Optional. Time interval (e.g., 'h1', 'd1'). Defaults to Financial Agent's internal default if None.
            days: Optional. Number of days to analyze. Defaults to Financial Agent's internal default if None.

        Returns:
            Historical analysis data as a JSON string.
        """
        query_parts = [f"Perform historical analysis for {symbol}"]
        if interval:
            query_parts.append(f"with interval {interval}")
        if days:
            query_parts.append(f"for the last {days} days" if interval else f" for {days} days") # Adjust phrasing slightly

        query_text = " ".join(query_parts)
        logger.info(f"Delegating historical analysis query to financial agent: {query_text}")
        
        # Create a structured dictionary for the agent input
        historical_analysis_request = {
            "query": query_text,
            "symbol": symbol,
            "action": "historical_analysis",
            "interval": interval,
            "days": days
        }
        
        # Call the specialized agent with the structured request
        return self._call_specialized_agent(
            agent_name="financial_data", 
            agent_input=historical_analysis_request,
            query_text=query_text,
            tool_context=None  # Don't pass the tool_context directly
        )

    # Updated fetch_news_sentiment with subreddit parameter
    def fetch_news_sentiment(self, company: str, subreddit: str, tool_context: ToolContext, timeframe: Optional[str] = "latest") -> str:
        """
        Fetch news and social media sentiment for a specific company/crypto from a specified subreddit using the Sentiment Analysis Agent.
        
        Args:
            company: The company/crypto ticker symbol or identifier (e.g., 'AAPL', 'GOOG', 'BTC').
            subreddit: The target subreddit name (e.g., 'Bitcoin', 'ethereum').
            tool_context: The context provided by the ADK framework.
            timeframe: Optional. Time period for sentiment analysis (e.g., 'last_week', 'last_month'). Defaults to 'latest'.
            
        Returns:
            Sentiment analysis results as a JSON string.
        """
        # Normalize subreddit name - remove 'r/' prefix if present
        if subreddit.startswith('r/'):
            subreddit = subreddit[2:]
            logger.info(f"Removed r/ prefix from subreddit name: {subreddit}")

        # Generate a session ID for this request to help with tracing and debugging
        session_id = str(uuid.uuid4())

        query_text = f"Get latest sentiment for {company} from subreddit {subreddit}"
        logger.info(f"Calling sentiment agent with query: {query_text}")
        
        # Create a proper agent_input dictionary
        sentiment_request = {
            "query": query_text,
            "company": company,
            "subreddit": subreddit,
            "timeframe": timeframe,
            "session_id": session_id
        }
        
        # Call the specialized agent with the structured request
        return self._call_specialized_agent(
            agent_name="sentiment_analysis", 
            agent_input=sentiment_request,
            query_text=query_text,
            tool_context=None  # Don't pass the tool_context directly
        )

    def analyze_competitors(self, company: str, metrics: List[str], tool_context: ToolContext) -> str:
        """
        Analyze competitors for a specific company based on selected metrics.
        
        Args:
            company: The company ticker symbol or identifier
            metrics: List of metrics to analyze (e.g., market_share, growth, profitability)
            
        Returns:
            Competitor analysis results
        """
        try:
            # Create a structured request
            query_text = f"Analyze competitors for {company} based on {', '.join(metrics)}"
            competitors_request = {
                "query": query_text,
                "company": company,
                "metrics": metrics,
                "action": "analyze_competitors"
            }
            
            # In a real implementation, this would call the specialized agent
            # return self._call_specialized_agent(
            #     agent_name="competitor_analysis", 
            #     agent_input=competitors_request,
            #     query_text=query_text,
            #     tool_context=None
            # )
            
            # For demonstration, we'll return mock data
            return json.dumps({
                "company": company,
                "metrics": metrics,
                "status": "Competitor analysis completed",
                "competitors": [
                    {"name": "Competitor A", "market_share": "25%", "growth": "12%"},
                    {"name": "Competitor B", "market_share": "18%", "growth": "8%"},
                    {"name": "Competitor C", "market_share": "15%", "growth": "20%"}
                ]
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    def generate_visualization(self, data_type: str, parameters: Dict[str, Any], tool_context: ToolContext) -> str:
        """
        Generate visual representations of data using the Visualization Agent.
        
        Args:
            data_type: Type of data to visualize (e.g., bar_chart, line_chart, pie_chart)
            parameters: Parameters for visualization generation including data for the plot
            tool_context: The context provided by the ADK framework
            
        Returns:
            JSON string containing the visualization data or error message
        """
        try:
            logger.info(f"Generating visualization of type: {data_type} with parameters: {parameters}")
            
            # Prepare visualization data based on data_type and parameters
            if data_type == "bar_chart":
                # Format data for bar chart visualization
                # Support both x_axis_data and x_axis_categories for backward compatibility
                labels = parameters.get("x_axis_data", parameters.get("x_axis_categories", []))
                values = parameters.get("y_axis_data", parameters.get("y_axis_values", []))
                x_label = parameters.get("x_axis_label", "X-Axis")
                y_label = parameters.get("y_axis_label", "Y-Axis")
                title = parameters.get("title", f"{y_label} by {x_label}")
                
                # Create plot description
                plot_description = f"Generate a {data_type} showing {y_label} for different {x_label}"
                
                # Create a visualization summary for the response
                vis_summary = {
                    "type": "bar_chart",
                    "title": title,
                    "x_axis": x_label,
                    "y_axis": y_label,
                    "data_points": [{"label": label, "value": value} for label, value in zip(labels, values)] if labels and values else [],
                    "message": f"A bar chart visualization has been created showing {y_label} for different {x_label}."
                }
                
                # Create data_json string with both formats for backward compatibility
                data_json = {
                    "labels": labels,
                    "values": values, 
                    "xlabel": x_label,
                    "ylabel": y_label,
                    "x_axis_data": labels,
                    "y_axis_data": values,
                    "x_axis_label": x_label,
                    "y_axis_label": y_label,
                    "title": title
                }
                
                # Convert data_json to string
                data_json_str = json.dumps(data_json)
                
            elif data_type == "line_chart":
                # Format data for line chart visualization
                dates = parameters.get("x_axis_data", parameters.get("x_axis_categories", []))
                values = parameters.get("y_axis_data", parameters.get("y_axis_values", []))
                x_label = parameters.get("x_axis_label", "Date")
                y_label = parameters.get("y_axis_label", "Value")
                title = parameters.get("title", f"{y_label} over {x_label}")
                
                # Create plot description
                plot_description = f"Generate a line chart showing {y_label} over {x_label}"
                
                # Create a visualization summary for the response
                vis_summary = {
                    "type": "line_chart",
                    "title": title,
                    "x_axis": x_label,
                    "y_axis": y_label,
                    "data_points": [{"date": date, "value": value} for date, value in zip(dates, values)] if dates and values else [],
                    "message": f"A line chart visualization has been created showing {y_label} over time."
                }
                
                # Create data_json string
                data_json = {
                    "labels": dates,
                    "values": values, 
                    "xlabel": x_label,
                    "ylabel": y_label,
                    "x_axis_data": dates,
                    "y_axis_data": values,
                    "x_axis_label": x_label,
                    "y_axis_label": y_label,
                    "title": title
                }
                data_json_str = json.dumps(data_json)
                
            else:
                # Default format
                title = parameters.get("title", f"{data_type.capitalize()} Visualization")
                plot_description = f"Generate a {data_type} titled '{title}'"
                vis_summary = {
                    "type": data_type,
                    "title": title,
                    "message": f"A {data_type} visualization has been created based on the provided data."
                }
                data_json_str = json.dumps(parameters)
            
            # Prepare request for the visualization agent
            visualization_request = {
                "plot_description": plot_description,
                "data_json": data_json_str
            }
            
            logger.info(f"Calling visualization agent with request: {str(visualization_request)[:200]}...")
            
            # Call the visualization agent
            agent_url = self.agent_urls.get("visualization", "http://localhost:8004")
            result = self._call_specialized_agent(
                agent_url=agent_url,
                agent_input=visualization_request,
                query_text=plot_description,
                tool_context=None  # Don't pass the tool_context directly
            )
            
            # Parse the result
            try:
                result_data = json.loads(result)
                
                # Extract any image data if present
                if "image_data" in result_data:
                    # Include the visualization summary and image data in the response
                    plot_id = result_data.get("image_data", {}).get("name", "unknown_plot.png")
                    return json.dumps({
                        "status": "success",
                        "message": "Visualization generated successfully. Note that this interface doesn't support direct image display.",
                        "visualization_summary": vis_summary,
                        "plot_id": plot_id
                    })
                # Handle case where status is success but no image data
                elif result_data.get("status") == "success":
                    # Include the visualization summary in the response
                    return json.dumps({
                        "status": "success",
                        "message": "Visualization generated successfully. Note that this interface doesn't support direct image display.",
                        "visualization_summary": vis_summary,
                        "plot_id": f"plot_{uuid.uuid4().hex}.png"
                    })
                else:
                    # Return the raw result if it doesn't match expected format
                    return result
                    
            except json.JSONDecodeError:
                logger.warning(f"Visualization agent returned non-JSON result: {result[:200]}...")
                # Try to provide a reasonable fallback
                return json.dumps({
                    "status": "success", 
                    "message": "Visualization generated successfully. Note that this interface doesn't support direct image display.",
                    "visualization_summary": vis_summary,
                    "plot_id": f"plot_{uuid.uuid4().hex}.png"
                })
                
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            logger.error(traceback.format_exc())
            return json.dumps({
                "status": "error", 
                "message": f"Failed to generate visualization: {str(e)}"
            })

    def get_analysis_template(self, template_type: str, tool_context: ToolContext) -> str:
        """
        Get standardized analysis templates.
        
        Args:
            template_type: Type of template to fetch (e.g., market_analysis, investment_thesis)
            
        Returns:
            Template for the specified analysis type
        """
        try:
            # In a real implementation, this would make an A2A call to the Prompt Templates Agent
            # For demonstration, we'll return mock data
            templates = {
                "market_analysis": "# Market Analysis Template\\n1. Industry Overview\\n2. Market Size and Growth\\n3. Key Trends\\n4. Competitive Landscape\\n5. Future Outlook",
                "investment_thesis": "# Investment Thesis Template\\n1. Company Overview\\n2. Financials\\n3. Competitive Position\\n4. Growth Drivers\\n5. Risks and Mitigations\\n6. Valuation and Recommendation"
            }
            
            # Create a structured request
            query_text = f"Get {template_type} template"
            template_request = {
                "query": query_text,
                "template_type": template_type,
                "action": "get_template"
            }
            
            # Call the specialized agent with the structured request or return mock data for now
            # return self._call_specialized_agent(
            #     agent_name="prompt_templates", 
            #     agent_input=template_request,
            #     query_text=query_text,
            #     tool_context=None
            # )
            
            # Just return mock data for now
            return json.dumps({
                "template_type": template_type,
                "status": "Template retrieved",
                "template": templates.get(template_type, "Template not found")
            })
        except Exception as e:
            return json.dumps({"error": str(e)})