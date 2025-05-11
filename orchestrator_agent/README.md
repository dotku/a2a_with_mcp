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

- `POST /`: Main JSON-RPC endpoint for A2A requests.
- `GET /.well-known/agent.json`: Returns the agent card/manifest.
- `GET /health`: Health check endpoint.
- `WebSocket /ws`: WebSocket endpoint for streaming responses.
- `POST /push/{task_id}`: Endpoint for receiving push notifications from delegated tasks.

## API Configuration

### Base URL
- **Default URL**: `http://localhost:8000/`
- **Environment Variable**: Set `HOST` and `PORT` to configure the server address
- **Usage**: The URL is used by clients to connect to the agent

### Capabilities
- **Streaming**: `true` - Agent supports real-time streaming of results
- **Push Notifications**: `true` - Agent supports HTTP callbacks for task updates
- **State Transition History**: `true` - Agent tracks and provides task state transitions

### Supported JSON-RPC Methods

| Method | Description |
|--------|-------------|
| `tasks/send` | Send a task to the orchestrator agent |
| `tasks/get` | Get information about an existing task |
| `tasks/cancel` | Cancel a running task |
| `tasks/sendSubscribe` | Create a task and subscribe to updates via WebSocket |
| `setPushNotificationConfig` | Configure push notifications for a task |
| `getPushNotificationConfig` | Get the push notification configuration for a task |

### Example Payloads

#### Example Request (tasks/send)

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
          "text": "Analyze the current sentiment for Bitcoin and provide price trends"
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

#### Example Response (tasks/send)

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

#### Example Response (tasks/get after completion)

```json
{
  "jsonrpc": "2.0",
  "id": "request-789",
  "result": {
    "id": "task-123",
    "sessionId": "session-456",
    "status": {
      "state": "COMPLETED",
      "timestamp": "2023-06-14T18:32:30.456Z",
      "message": {
        "role": "assistant",
        "parts": [
          {
            "type": "text",
            "text": "Based on my analysis of Bitcoin, the current sentiment is generally positive with 65% positive mentions on social media. The price has shown an upward trend of 8.5% over the past week."
          }
        ]
      }
    },
    "artifacts": [
      {
        "name": "bitcoin-analysis",
        "description": "Bitcoin sentiment and price analysis",
        "parts": [
          {
            "type": "text",
            "text": "{ \"sentiment\": \"positive\", \"sentiment_score\": 0.65, \"price_trend\": \"up\", \"price_change_7d\": \"8.5%\" }"
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
            "text": "Analyze the current sentiment for Bitcoin and provide price trends"
          }
        ]
      },
      {
        "role": "assistant",
        "parts": [
          {
            "type": "text",
            "text": "Based on my analysis of Bitcoin, the current sentiment is generally positive with 65% positive mentions on social media. The price has shown an upward trend of 8.5% over the past week."
          }
        ]
      }
    ]
  }
}
``` 