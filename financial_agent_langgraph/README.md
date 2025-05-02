# Financial Analysis Agent (LangGraph)

This agent provides financial analysis capabilities for the Multi-Agent Market Research & Financial Analysis System. It's built using LangGraph, follows the Agent-to-Agent (A2A) protocol for communication, and integrates with multiple MCP (Model Context Protocol) servers for data retrieval.

## Features

- Executes SQL queries against a financial database (via Postgres MCP server).
- Retrieves current and historical stock information (via Financial Agent's own tools).
- Performs detailed cryptocurrency market analysis (via Crypto Price MCP server).
- Provides historical cryptocurrency price analysis (via Crypto Price MCP server).
- Synthesizes information from multiple sources using LangGraph.

## Architecture

- **LangGraph Core:** Uses LangGraph for defining the agent's reasoning workflow.
- **Multi-MCP Integration:** Connects to:
    - A **Postgres MCP server** (assumed to be running at `MCP-servers/postgres_mcp.py`) for database interactions (e.g., `query` tool).
    - The **`mcp-crypto-price` MCP server** (run via `npx`) for external crypto market data (`get-market-analysis`, `get-historical-analysis` tools).
- **A2A Compliance:** Exposes a FastAPI server compatible with the A2A protocol.
- **Tool Filtering:** Intelligently filters tools fetched from MCP servers to provide the most relevant capabilities to the agent's LLM (e.g., uses the database `query` for current prices, uses crypto MCP for analysis).

## Prerequisites

1.  Python 3.10+
2.  **Node.js and npm/npx:** Required to run the `mcp-crypto-price` server. Ensure `npx` is available in your PATH.
3.  **Postgres MCP Server:** The agent expects the Postgres MCP server script to be available at `MCP-servers/postgres_mcp.py` relative to the project root where the agent is launched. (Adjust path in `agent.py` if necessary).

## Setup & Running

1.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Set Environment Variables:**
    *   `OPENAI_API_KEY`: Your OpenAI API key for the LLM.
    *   `COINCAP_API_KEY`: **(Recommended)** Your CoinCap API key for the `mcp-crypto-price` server. Get one from [pro.coincap.io/dashboard](https://pro.coincap.io/dashboard). If not provided, the crypto server might use the v2 API (being sunset) or have rate limits.
    ```bash
    export OPENAI_API_KEY="your_openai_key"
    export COINCAP_API_KEY="your_coincap_key" 
    ```
3.  **Run the Server:**
    ```bash
    python -m financial_agent_langgraph
    ```
    This command starts the FastAPI server (default port 8001) and automatically launches the two required MCP server subprocesses (Postgres and `mcp-crypto-price` via `npx`). Ensure `npx` can find `mcp-crypto-price` (install globally with `npm install -g mcp-crypto-price` if needed, although `npx` often handles temporary installs).

## API Endpoints

- `POST /`: Main JSON-RPC endpoint for A2A requests.
- `GET /.well-known/agent.json`: Returns the agent metadata. (Ensure this is implemented/updated in `server.py`).
- `GET /health`: Health check endpoint. (Ensure this is implemented in `server.py`).
- (WebSocket endpoint if implemented).

## A2A Methods Supported

(List relevant methods like `tasks/send`, `tasks/get` etc., as implemented in `server.py`).

## Project Structure

- `agent.py`: LangGraph financial analysis workflow, MCP client initialization, tool filtering.
- `task_manager.py`: Manages task lifecycle and state.
- `server.py`: FastAPI server for A2A endpoints.
- `__main__.py`: Entry point for running the server.
- `common/`: Shared types and utilities.
- `MCP-servers/postgres_mcp.py`: (External) Assumed location of the Postgres MCP server script.
- `mcp-crypto-price`: (External) Node.js package run via `npx`.

## Future Enhancements

- Integration with PostgreSQL MCP server for real data
- More sophisticated financial analysis tools
- Support for additional financial data sources
- Enhanced visualization capabilities

## Project Structure

- `agent.py`: LangGraph financial analysis workflow
- `task_manager.py`: Manages task lifecycle and state
- `server.py`: FastAPI server for A2A endpoints
- `__main__.py`: Entry point for running the server
- `common/`: Shared types and utilities 