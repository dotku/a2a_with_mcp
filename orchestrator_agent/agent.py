import json
import uuid
from typing import Any, AsyncIterable, Dict, List, Optional
import traceback
import logging
import time
import httpx

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
    def _call_specialized_agent(self, agent_name=None, agent_url=None, agent_input=None, query_text=None, tool_context=None, max_retries=1) -> str:
        """
        Calls a specialized agent, sends a task, and polls for the result.
        
        Args:
            agent_name: Name of the agent to call (used to look up URL)
            agent_url: Direct URL to the agent (alternative to agent_name)
            agent_input: Input to send to the agent
            query_text: Backward compatibility for older code
            tool_context: Optional context from tool invocation
            max_retries: Number of retries to attempt if a call fails
            
        Returns:
            JSON string with agent response or error
        """
        try:
            if not agent_url and not agent_name:
                return json.dumps({
                    "status": "error", 
                    "message": "Either agent_name or agent_url must be provided"
                })
            
            # Get agent URL from mapping if name is provided
            if not agent_url:
                agent_url = self.agent_urls.get(agent_name)
                if not agent_url:
                    return json.dumps({
                        "status": "error", 
                        "message": f"No URL found for agent: {agent_name}"
                    })
            
            # Create task ID and session ID
            task_id = f"task-{uuid.uuid4()}"
            session_id = f"{uuid.uuid4()}"
            
            # If legacy query_text is provided, wrap it (backward compatibility)
            if query_text and not agent_input:
                agent_input = {"query": query_text}
            
            # Prepare the agent request payload
            agent_request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tasks/send",
                "params": {
                    "id": task_id,
                    "sessionId": session_id,
                    "acceptedOutputModes": ["text"],
                    "message": {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": json.dumps(agent_input) if isinstance(agent_input, dict) else agent_input
                            }
                        ]
                    }
                }
            }
            
            logger.info(f"Sending task/send request to Specialized Agent: {agent_request}")
            
            # Make the initial request to send the task
            attempts = 0
            error_message = None
            response_data = None
            
            while attempts < max_retries:
                attempts += 1
                try:
                    response = httpx.post(agent_url, json=agent_request, timeout=60.0)
                    response.raise_for_status()  # Raise exception for non-200 responses
                    response_data = response.json()
                    logger.info(f"Received task/send response from Specialized Agent: {response_data}")
                    
                    # Check if task was completed immediately
                    if (
                        "result" in response_data 
                        and isinstance(response_data["result"], dict)
                        and "status" in response_data["result"]
                        and isinstance(response_data["result"]["status"], dict)
                        and "state" in response_data["result"]["status"]
                    ):
                        task_state = response_data["result"]["status"]["state"]
                        
                        if task_state == "completed":
                            logger.info(f"Task {task_id} reported as COMPLETED in initial response. Extracting result.")
                            # Extract the result from the response
                            return self._extract_result_from_task_data(response_data["result"], task_id)
                        elif task_state == "error":
                            error_message = "Task reported error state"
                            logger.error(f"Task {task_id} reported ERROR state: {response_data}")
                            # Try again if we haven't reached max retries
                            continue
                            
                    # Task requires polling - unfortunately this isn't implemented yet
                    # In reality, we need to poll using tasks/get until completion
                    return json.dumps({
                        "status": "error",
                        "message": "Task requires polling which is not yet implemented"
                    })
                    
                except httpx.RequestError as e:
                    error_message = f"Request error calling {agent_url}: {str(e)}"
                    logger.error(error_message)
                    # Wait before retrying
                    time.sleep(2)
                except httpx.HTTPStatusError as e:
                    error_message = f"HTTP error calling {agent_url}: {str(e)}"
                    logger.error(error_message)
                    # Wait before retrying
                    time.sleep(2)
                except Exception as e:
                    error_message = f"Error calling {agent_url}: {str(e)}"
                    logger.error(error_message)
                    logger.error(traceback.format_exc())
                    # Wait before retrying
                    time.sleep(2)
                    
            # If we get here, all retries failed
            if error_message:
                return json.dumps({"status": "error", "message": error_message})
            else:
                return json.dumps({"status": "error", "message": "Unknown error during specialized agent call"})
                
        except Exception as e:
            logger.error(f"Error calling specialized agent: {str(e)}")
            logger.error(traceback.format_exc())
            return json.dumps({"status": "error", "message": f"Internal error: {str(e)}"})

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
        return self._call_specialized_agent("financial_data", query_text, tool_context)

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
        else:
            # Proceed with historical data request
            query_text = f"Fetch stock price history for {ticker} from {start_date} to {end_date}"
            
        return self._call_specialized_agent("financial_data", query_text, tool_context)

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
        # The financial agent's LLM should recognize this based on its SYSTEM_INSTRUCTION
        query_text = f"Run SQL query: {query}"
        return self._call_specialized_agent("financial_data", query_text, tool_context)

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
        return self._call_specialized_agent("financial_data", query_text, tool_context)

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
        return self._call_specialized_agent("financial_data", query_text, tool_context)

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
        # Construct the query based on the timeframe and subreddit
        if timeframe == "latest":
            query_text = f"Get latest sentiment for {company} from subreddit {subreddit}"
        else:
            query_text = f"Get sentiment for {company} from subreddit {subreddit} over the {timeframe}"
            
        logger.info(f"Calling sentiment agent with query: {query_text}")
        # Call the specialized agent instead of returning mock data
        return self._call_specialized_agent("sentiment_analysis", query_text, tool_context)

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
            # In a real implementation, this would make an A2A call to the Competitor Analysis Agent
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
                agent_input=json.dumps(visualization_request),
                tool_context=tool_context
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
            return json.dumps({
                "template_type": template_type,
                "status": "Template retrieved",
                "template": templates.get(template_type, "Template not found")
            })
        except Exception as e:
            return json.dumps({"error": str(e)})