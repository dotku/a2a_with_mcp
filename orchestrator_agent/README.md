# Orchestrator Agent

This agent acts as the central coordinator for the Multi-Agent Market Research & Financial Analysis System. It receives user requests, breaks them down into subtasks, and delegates these tasks to specialized agents via the Agent-to-Agent (A2A) protocol.

## Architecture

- **Core Logic:** Built using the Google Agent Development Kit (ADK) and the `gemini-2.0-flash-lite` model.
- **Task Delegation:** Identifies the appropriate specialized agent based on the user request and delegates tasks using defined tools.
- **A2A Communication:** Interacts with specialized agents using JSON-RPC over HTTP, following the A2A protocol for sending tasks and polling for results.
- **Agent Discovery:** Relies on pre-configured URLs for specialized agents.

## Specialized Agents Coordinated

The orchestrator is configured to interact with the following agents:

1.  **Financial Data Agent (`http://localhost:8001`):**
    *   Retrieves financial statements.
    *   Fetches current or historical stock prices.
    *   Executes SQL queries on a financial database.
    *   **NEW:** Performs detailed cryptocurrency market analysis.
    *   **NEW:** Provides historical cryptocurrency price analysis.
2.  **Sentiment Analysis Agent (`http://localhost:10000`):**
    *   Analyzes news and social media sentiment for specific companies or cryptocurrencies based on subreddit mappings.
3.  **Competitor Analysis Agent (`http://localhost:8003`):**
    *   Analyzes competitor data and market positioning. (Currently uses mock data).
4.  **Visualization Agent (`http://localhost:8004`):**
    *   Creates visual representations of data. (Currently uses mock data).
5.  **Prompt Templates Agent (`http://localhost:8005`):**
    *   Provides standardized analysis templates. (Currently uses mock data).

## Orchestrator Tools

The orchestrator uses the following internal tools to delegate tasks:

- `fetch_financial_statements`: Delegates to Financial Agent.
- `fetch_stock_price_history`: Delegates to Financial Agent.
- `run_sql_query`: Delegates to Financial Agent.
- `get_crypto_market_analysis`: **(NEW)** Delegates crypto market analysis to Financial Agent.
- `get_crypto_historical_analysis`: **(NEW)** Delegates crypto historical analysis to Financial Agent.
- `fetch_news_sentiment`: Delegates to Sentiment Analysis Agent.
- `analyze_competitors`: Delegates to Competitor Analysis Agent.
- `generate_visualization`: Delegates to Visualization Agent.
- `get_analysis_template`: Delegates to Prompt Templates Agent.

## Running the Orchestrator

1.  Ensure you have Python 3.10+ installed.
2.  Install dependencies (likely shared with other agents or defined in a root `requirements.txt`).
3.  Set any required environment variables (e.g., Google API keys for the underlying ADK/Gemini model).
4.  Run the server:
    ```bash
    python -m orchestrator_agent 
    ```
    The server typically starts on port 8000.

## API Endpoints

- `POST /`: Main endpoint expected by the UI or calling application.
- (Add other relevant endpoints if exposed, e.g., health check). 