# Multi-Agent Market Research & Financial Analysis System

This project demonstrates a sophisticated multi-agent system designed for market research and financial analysis. It combines the power of large language models (LLMs), specialized agents, and external data sources via the Model Context Protocol (MCP).

## Overview

The system features an **Orchestrator Agent** that receives user requests and delegates subtasks to specialized agents. The key specialized agent is the **Financial Analysis Agent**, which has been enhanced to connect to multiple data sources.

## Architecture

1.  **User Interface (UI):** (Assumed) Interacts with the Orchestrator Agent.
2.  **Orchestrator Agent (`orchestrator_agent`):**
    *   Built with Google ADK.
    *   Receives user requests.
    *   Uses an LLM (Gemini Flash) to determine the required subtasks.
    *   Delegates tasks to specialized agents via A2A calls using specific tools.
    *   Synthesizes results from specialized agents into a final response.
3.  **Specialized Agents:**
    *   **Financial Analysis Agent (`financial_agent_langgraph`):**
        *   Built with LangGraph.
        *   Connects to a **Postgres MCP server** for internal database queries (current prices, company info).
        *   Connects to the **`mcp-crypto-price` MCP server** (Node.js) for external crypto market/historical analysis.
        *   Exposes its capabilities via an A2A-compliant FastAPI server.
    *   **Sentiment Analysis Agent (`sentiment_analysis_agent`):**
        *   Built with CrewAI (in this example).
        *   Connects to an **MCP Reddit server** to fetch posts.
        *   Analyzes sentiment related to specific companies/cryptocurrencies.
        *   Exposes its capabilities via an A2A-compliant FastAPI server.
    *   **Other Agents (Competitor Analysis, Visualization, Templates):**
        *   Placeholder agents demonstrating potential extensions (currently return mock data in the orchestrator).
4.  **MCP Servers:**
    *   **Postgres MCP Server:** (Assumed `MCP-servers/postgres_mcp.py`) Provides tools to interact with a financial PostgreSQL database.
    *   **Crypto Price MCP Server:** (`mcp-crypto-price` Node.js package) Provides tools (`get-market-analysis`, `get-historical-analysis`) using the CoinCap API.
    *   **Reddit MCP Server:** (Located within `sentiment_analysis_agent`) Provides tools to fetch Reddit data.

## Key Features & Concepts

- **Multi-Agent Collaboration:** Demonstrates how an orchestrator can manage and leverage multiple specialized agents.
- **A2A Protocol:** Uses a standardized protocol for inter-agent communication (JSON-RPC based).
- **Model Context Protocol (MCP):** Enables agents (like the Financial Agent) to securely and reliably use external tools and data sources (Postgres DB, CoinCap API via `mcp-crypto-price`, Reddit API) served via MCP.
- **Multi-Backend MCP:** The Financial Agent showcases connecting to multiple, different MCP servers simultaneously.
- **LangGraph & CrewAI:** Utilizes different frameworks for building the core logic of specialized agents.

## Running the System

1.  **Prerequisites:** Python 3.10+, Node.js/npx, PostgreSQL database, API Keys (OpenAI, CoinCap, potentially Reddit/PRAW).
2.  **Setup:**
    *   Clone the repository.
    *   Install Python dependencies (`pip install -r requirements.txt` - ensure a consolidated requirements file exists or install per-agent).
    *   Set required environment variables (see individual agent READMEs).
    *   Ensure MCP server scripts/packages are accessible (`MCP-servers/postgres_mcp.py`, `mcp-crypto-price` via npx).
3.  **Launch Agents (in separate terminals):**
    *   `python -m financial_agent_langgraph` (Handles its MCP servers internally)
    *   `uvicorn sentiment_analysis_agent.agent_server:app --reload --port 10000` (Or however the sentiment agent is run)
    *   `python -m orchestrator_agent`
    *   (Launch other specialized agents if implemented).
4.  **Interact:** Use the UI or send requests directly to the Orchestrator Agent (default port 8000).

## Project Structure

- `orchestrator_agent/`: Code for the central orchestrator.
- `financial_agent_langgraph/`: Code for the LangGraph-based financial agent.
- `sentiment_analysis_agent/`: Code for the CrewAI-based sentiment agent.
- `MCP-servers/`: (Example location) Contains MCP server implementations (e.g., `postgres_mcp.py`).
- `README.md`: This file.
- (Potentially `requirements.txt`, `.env.example`, etc.)

*(See individual agent directories for their specific README files)*

## Requirements

- Python 3.9 or higher
- PostgreSQL database server
- OpenAI API key (for GPT models)

## Installation

1. Clone this repository
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your PostgreSQL database:
   ```sql
   -- Create the companies table
   CREATE TABLE companies (
       ticker TEXT PRIMARY KEY,
       name TEXT NOT NULL,
       sector TEXT,
       industry TEXT
   );

   -- Create the stocks table
   CREATE TABLE stocks (
       id SERIAL PRIMARY KEY,
       symbol TEXT NOT NULL REFERENCES companies(ticker),
       price NUMERIC(10, 2),
       volume INTEGER,
       timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   -- Create the financial_metrics table
   CREATE TABLE financial_metrics (
       id SERIAL PRIMARY KEY,
       ticker TEXT NOT NULL REFERENCES companies(ticker),
       metric TEXT NOT NULL,
       value NUMERIC(15, 2),
       period TEXT
   );

   -- Create the financial_snapshot table
   CREATE TABLE financial_snapshot (
       id SERIAL PRIMARY KEY,
       data JSONB,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   -- Insert sample data as needed
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Configuration

1. Update the PostgreSQL connection settings in `MCP-servers/postgres_mcp.py`:
   ```python
   db_url = "postgresql://postgres:password@localhost:5432/testdb"
   ```
   Replace with your database credentials.

## Usage

### 1. Start the MCP Postgres Server

```bash
python3 MCP-servers/postgres_mcp.py
```

### 2. Run the Financial Agent

You can run the agent in several ways:

#### Testing Direct Queries

```bash
python test_financial_agent.py
```

#### Testing MCP Connection

```bash
python test_mcp_connection.py
```

#### Testing Async MCP Tool Invocation

```bash
python test_async_mcp.py
```

### 3. Example Queries

You can ask the agent questions like:

- "What's the current stock price for AAPL?"
- "Show me financial metrics for Microsoft"
- "Compare the P/E ratios of tech companies"
- "What are the latest market trends from the financial snapshot?"

## MCP Tools Available

The agent can access the following MCP tools:

1. **query**: Run arbitrary SQL queries against the database
   ```
   Example: SELECT * FROM stocks WHERE symbol = 'AAPL'
   ```

2. **fetch_financial_snapshot**: Get the latest financial snapshot data as a JSON object
   ```
   This returns a comprehensive market overview from the financial_snapshot table
   ```

## Advanced Configuration

### Event Loop Management

The agent implements a shared event loop system for all MCP operations to prevent "Event loop is closed" errors during async operations. This is handled in the `get_mcp_event_loop()` function.

### Error Handling

The system includes robust error handling with fallback mechanisms for:
- Database connection issues
- MCP server communication failures
- Async event loop errors

## Troubleshooting

1. **Event Loop Errors**: If you experience "Event loop is closed" errors, restart the MCP server and the agent.

2. **Database Connection Issues**: Ensure your PostgreSQL server is running and the credentials in `postgres_mcp.py` are correct.

3. **Tool Execution Failures**: Check the logs in `financial_agent.log` for detailed error messages.

## License

[Your License Information]

## Acknowledgments

- LangChain and LangGraph for the agent framework
- The Model Context Protocol (MCP) for standardizing tool access

## Environment Variables

The system requires various API keys and configuration settings to be provided as environment variables. Create a `.env` file in the root directory with the following variables:

```
# OpenAI API for financial agent
OPENAI_API_KEY=your_openai_api_key_here

# Google API for sentiment and orchestrator agents
GOOGLE_API_KEY=your_google_api_key_here

# CoinCap API for cryptocurrency data
COINCAP_API_KEY=your_coincap_api_key_here

# Database configuration for financial agent
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password_here
DB_NAME=financial_db

# Agent ports (optional, defaults shown)
SENTIMENT_AGENT_PORT=10000
FINANCIAL_AGENT_PORT=10001
ORCHESTRATOR_PORT=8000
```

Each agent has specific environment variable requirements:

1. **Financial Agent**: Requires `OPENAI_API_KEY`, database configuration, and `COINCAP_API_KEY`
2. **Sentiment Agent**: Requires `GOOGLE_API_KEY`
3. **Orchestrator Agent**: Requires `GOOGLE_API_KEY`
4. **Visualization Agent**: Requires `GOOGLE_API_KEY`

Environment variables can also be set directly in your shell if you prefer not to use a `.env` file. 