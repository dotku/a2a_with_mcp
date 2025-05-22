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
    *   `FIN_AGENT_BASE_URL`: Base URL where the agent is hosted. Used in the agent card for discovery. Defaults to "http://localhost:8001/" if not specified.
    ```bash
    export OPENAI_API_KEY="your_openai_key"
    export COINCAP_API_KEY="your_coincap_key" 
    export FIN_AGENT_BASE_URL="https://your-agent-host.example.com/" 
    ```
3.  **Run the Server:**
    ```bash
   python -m financial_agent_langgraph
   ```
    This command starts the FastAPI server (default port 8001) and automatically launches the two required MCP server subprocesses (Postgres and `mcp-crypto-price` via `npx`). Ensure `npx` can find `mcp-crypto-price` (install globally with `npm install -g mcp-crypto-price` if needed, although `npx` often handles temporary installs).

## API Configuration

### Base URL
- **Environment Variable**: `FIN_AGENT_BASE_URL`
- **Default Value**: `http://localhost:8001/`
- **Usage**: Set this environment variable to specify where the agent is hosted

### Capabilities
- **Streaming**: `true` - Agent supports real-time streaming of results
- **Push Notifications**: `true` - Agent can send HTTP callbacks for task updates
- **State Transition History**: `true` - Agent maintains and provides task state history

## API Endpoints

- `POST /`: Main JSON-RPC endpoint for A2A requests.
- `GET /.well-known/agent.json`: Returns the agent metadata.
- `GET /health`: Health check endpoint.
- `WebSocket /ws`: WebSocket endpoint for streaming responses.

## Supported JSON-RPC Methods

| Method | Description |
|--------|-------------|
| `tasks/send` | Send a task to the financial agent for processing |
| `tasks/get` | Get information about an existing task |
| `tasks/cancel` | Cancel a running task |
| `tasks/sendSubscribe` | Create a task and subscribe to updates via WebSocket |
| `setPushNotificationConfig` | Configure push notifications for a task |
| `getPushNotificationConfig` | Get the push notification configuration for a task |

## Example Payloads

### Example Request (tasks/send)

```json
{
  "jsonrpc": "2.0",
  "id": "request-123",
  "method": "tasks/send",
  "params": {
    "id": "task-123",
    "sessionId": "session-456",
    "message": {
      "role": "user",
      "parts": [
        {
          "type": "text",
          "text": "What is the current P/E ratio for Apple?"
        }
      ]
    },
    "pushNotification": {
      "url": "https://your-callback-service.example.com/webhooks",
      "token": "your-auth-token"
    }
  }
}
```

### Example Response (tasks/send)

```json
{
  "jsonrpc": "2.0",
  "id": "request-123",
  "result": {
    "id": "task-123",
    "sessionId": "session-456",
    "status": {
      "state": "SUBMITTED",
      "timestamp": "2023-06-14T18:30:45.123Z"
    }
  }
}
```

### Example Response (tasks/get after completion)

```json
{
  "jsonrpc": "2.0",
  "id": "request-789",
  "result": {
    "id": "task-123",
    "sessionId": "session-456",
    "status": {
      "state": "COMPLETED",
      "timestamp": "2023-06-14T18:31:15.456Z",
      "message": {
        "role": "assistant",
        "parts": [
          {
            "type": "text",
            "text": "As of the latest data, Apple (AAPL) has a P/E ratio of approximately 30.25."
          }
        ]
      }
    },
    "artifacts": [
      {
        "name": "financial-metrics",
        "description": "Financial metrics for Apple",
        "parts": [
          {
            "type": "text",
            "text": "{ \"symbol\": \"AAPL\", \"pe_ratio\": 30.25, \"market_cap\": \"2.5T\", \"price\": 182.63 }"
          }
        ]
      }
    ],
    "history": [
      {
        "role": "user",
        "parts": [
          {
            "type": "text",
            "text": "What is the current P/E ratio for Apple?"
          }
        ]
      },
      {
        "role": "assistant",
        "parts": [
          {
            "type": "text",
            "text": "As of the latest data, Apple (AAPL) has a P/E ratio of approximately 30.25."
          }
        ]
      }
    ]
  }
}
```

## Project Structure

- `agent.py`: LangGraph financial analysis workflow
- `task_manager.py`: Manages task lifecycle and state
- `server.py`: FastAPI server for A2A endpoints
- `__main__.py`: Entry point for running the server
- `common/`: Shared types and utilities

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