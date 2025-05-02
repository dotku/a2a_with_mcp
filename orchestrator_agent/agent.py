import json
import uuid
from typing import Any, AsyncIterable, Dict, List, Optional
import traceback
import logging
import time

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

import httpx

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
                    "1. Financial Data Agent: For retrieving financial statements, stock price history, and running SQL queries on financial data.\n"
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
                    "You can use the following tools to interact with specialized agents:\n"
                    "- fetch_financial_statements: To request financial statements (income, balance sheet, cash flow) for a company and year.\n"
                    "- fetch_stock_price_history: To request historical stock prices for a ticker symbol within an optional date range. Omit dates for current price.\n"
                    "- run_sql_query: To execute a read-only SQL query (SELECT or WITH) on the financial database.\n"
                    "- fetch_news_sentiment: To analyze news and social media sentiment for a specific company/crypto and subreddit. Determine the correct subreddit using the mapping above. If the user asks for the 'latest' sentiment, use the default timeframe ('latest').\n"
                    "- analyze_competitors: To gather competitor information.\n"
                    "- generate_visualization: To create visual representations.\n"
                    "- get_analysis_template: To obtain standardized analysis templates.\n"
                ),
                # Updated tools list
                tools=[
                    self.fetch_financial_statements, # Renamed
                    self.fetch_stock_price_history, # Added
                    self.run_sql_query,             # Added
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
    def _call_specialized_agent(self, agent_name: str, query_text: str, tool_context: Optional[ToolContext]) -> str:
        """Calls a specialized agent, sends a task, and polls for the result."""
        try:
            agent_url = self.agent_urls.get(agent_name)
            if not agent_url:
                return json.dumps({"error": f"{agent_name.replace('_', ' ').title()} agent URL not configured"})

            session_id = str(uuid.uuid4())
            initial_task_id = f"task-{uuid.uuid4()}"
            
            send_request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tasks/send",
                "params": {
                    "id": initial_task_id,
                    "sessionId": session_id,
                    "acceptedOutputModes": ["text"],
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": query_text}]
                    }
                }
            }
            
            logger.info(f"Sending task/send request to {agent_name.replace('_', ' ').title()} Agent: {send_request}")
            
            with httpx.Client() as client:
                response = client.post(agent_url, json=send_request, timeout=60.0)
                response.raise_for_status()
                send_response = response.json()
                logger.info(f"Received task/send response from {agent_name.replace('_', ' ').title()} Agent: {send_response}")

                # Check the result structure and initial task state
                if "result" not in send_response or not isinstance(send_response.get("result"), dict):
                    error_msg = "Invalid response format from agent"
                    if "error" in send_response and send_response.get("error"):
                         error_obj = send_response["error"]
                         error_msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
                    logger.error(f"{error_msg} from {agent_name.replace('_', ' ').title()} Agent. Response: {send_response}")
                    return json.dumps({"error": error_msg})
                
                task_data = send_response["result"]
                task_id = task_data.get("id", initial_task_id) # Use actual ID if provided
                
                if "status" not in task_data or not isinstance(task_data.get("status"), dict) or "state" not in task_data["status"]:
                    error_msg = "Missing or invalid status in initial task response"
                    logger.error(f"{error_msg} from {agent_name.replace('_', ' ').title()} Agent. Response: {send_response}")
                    return json.dumps({"error": error_msg})

                initial_state = task_data["status"]["state"]
                
                # --- Handle immediate completion or failure ---
                if initial_state == "completed":
                    logger.info(f"Task {task_id} reported as COMPLETED in initial response. Extracting result.")
                    return self._extract_result_from_task_data(task_data, task_id)
                
                elif initial_state == "failed":
                    logger.error(f"Task {task_id} reported as FAILED in initial response.")
                    error_message = self._extract_error_from_task_data(task_data)
                    return json.dumps({"error": f"{agent_name.replace('_', ' ').title()} agent task failed: {error_message}"})
                
                elif initial_state not in ["submitted", "working"]:
                     # If state is not submitted/working and also not completed/failed, it's unexpected
                     error_msg = f"Task {task_id} started in unexpected state: {initial_state}"
                     logger.error(f"{error_msg} from {agent_name.replace('_', ' ').title()} Agent. Response: {send_response}")
                     return json.dumps({"error": error_msg})
                 
                # --- Proceed with polling only if state is submitted or working ---
                logger.info(f"Task {task_id} state is {initial_state}. Starting polling.")
                
                start_time = time.time()
                timeout_seconds = 60 # Adjust timeout as needed
                poll_interval_seconds = 2 # Adjust interval as needed

                while time.time() - start_time < timeout_seconds:
                    time.sleep(poll_interval_seconds)
                    
                    get_request = {
                        "jsonrpc": "2.0",
                        "id": str(uuid.uuid4()),
                        "method": "tasks/get",
                        "params": {"id": task_id}
                    }
                    
                    logger.debug(f"Polling task {task_id} status...")
                    get_response_obj = client.post(agent_url, json=get_request, timeout=30.0)
                    get_response_obj.raise_for_status()
                    get_response = get_response_obj.json()
                    logger.debug(f"Received task/get response: {get_response}")

                    if "result" in get_response and isinstance(get_response.get("result"), dict):
                        polled_task_data = get_response["result"]
                        if "status" in polled_task_data and isinstance(polled_task_data.get("status"), dict):
                            task_status = polled_task_data["status"]["state"]
                            
                            if task_status == "completed":
                                logger.info(f"Polling found task {task_id} completed.")
                                return self._extract_result_from_task_data(polled_task_data, task_id)

                            elif task_status == "failed":
                                logger.error(f"Polling found task {task_id} failed.")
                                error_message = self._extract_error_from_task_data(polled_task_data)
                                return json.dumps({"error": f"{agent_name.replace('_', ' ').title()} agent task failed: {error_message}"})

                            elif task_status in ["submitted", "working"]:
                                logger.debug(f"Task {task_id} still in state: {task_status}. Continuing polling.")
                                # Continue loop
                            else:
                                logger.warning(f"Task {task_id} in unexpected state during poll: {task_status}. Stopping poll.")
                                return json.dumps({"error": f"Task ended in unexpected state: {task_status}"})
                        else:
                             logger.warning(f"Polling response for task {task_id} missing status information: {get_response}")
                             # Optionally wait and retry, or return error after a few attempts
                             # For now, continue polling hoping the next response is valid
                    elif "error" in get_response and get_response["error"]:
                        # Handle errors from the tasks/get call itself (e.g., task not found)
                        error_obj = get_response["error"]
                        error_details = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
                        logger.error(f"Error polling task {task_id}: {error_details}")
                        return json.dumps({"error": f"Error retrieving task status: {error_details}"})
                    else:
                        logger.warning(f"Unexpected polling response format for task {task_id}: {get_response}")
                        # Continue polling hoping the next response is valid
                
                # If the loop finishes without completion
                logger.error(f"Polling timed out for task {task_id} after {timeout_seconds} seconds.")
                return json.dumps({"error": f"Timed out waiting for result from {agent_name.replace('_', ' ').title()} agent"})

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling {agent_name.replace('_', ' ').title()} Agent: {e.response.status_code} - {e.response.text}")
            return json.dumps({"error": f"HTTP error calling {agent_name.replace('_', ' ').title()} Agent: {e.response.status_code}", "details": e.response.text})
        except httpx.RequestError as e:
            logger.error(f"Request error calling {agent_name.replace('_', ' ').title()} Agent: {e}")
            return json.dumps({"error": f"Could not connect to {agent_name.replace('_', ' ').title()} Agent: {e}"})
        except Exception as e:
            logger.error(f"Error in _call_specialized_agent ({agent_name}): {str(e)}")
            logger.error(traceback.format_exc())
            return json.dumps({"error": f"An unexpected error occurred while calling {agent_name}: {str(e)}"})

    # --- Helper methods to extract data from task results ---
    def _extract_result_from_task_data(self, task_data: Dict, task_id: str) -> str:
        """Extracts the primary result text from completed task data."""
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
            logger.info(f"Extracted result from artifact for task {task_id}: {result_text[:200]}...")
            # Attempt to parse JSON, otherwise return text
            try:
                # Important: Ensure the specialized agent actually returns JSON if needed.
                # If it's just text, this parse will fail, and we'll wrap it.
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
            logger.info(f"Extracted result from status message for task {task_id}: {result_text[:200]}...")
            # Assume status message text is not meant to be JSON, wrap it
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
        Generate visual representations of data.
        
        Args:
            data_type: Type of data to visualize (e.g., financial_metrics, market_share, sentiment)
            parameters: Parameters for visualization generation
            
        Returns:
            URL or base64-encoded image of the generated chart
        """
        try:
            # In a real implementation, this would make an A2A call to the Visualization Agent
            # For demonstration, we'll return mock data
            return json.dumps({
                "data_type": data_type,
                "parameters": parameters,
                "status": "Visualization generated",
                "chart_url": "https://example.com/charts/abc123",
                "chart_type": parameters.get("chart_type", "bar")
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

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