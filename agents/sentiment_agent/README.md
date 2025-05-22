# Bitcoin Sentiment Analysis Agent

This agent analyzes sentiment for Bitcoin discussions on Reddit using the CrewAI framework. It implements the Agent-to-Agent (A2A) protocol for interoperability with other A2A-compliant agents and orchestrators.

## Features

- Bitcoin sentiment analysis from Reddit data
- Full A2A protocol compatibility for agent discovery and task management
- Integration with the A2A UI framework
- Synchronous processing (non-streaming)

## Setup and Running

### 1. Install Dependencies

Make sure you have all required dependencies installed:

```bash
pip install crewai fastapi uvicorn httpx pydantic python-dotenv
```

### 2. Set Environment Variables

You can set environment variables in two ways:

#### Option 1: Using a .env file (recommended)

Create a `.env` file in either the root directory or inside the `sentiment_analysis_agent` directory:

```
# .env file
GOOGLE_API_KEY=your_google_api_key_here
SENTIMENT_AGENT_PORT=10000  # Optional, defaults to 10000
```

The server will automatically load variables from this file when starting.

#### Option 2: Set environment variables directly

```bash
export GOOGLE_API_KEY=your_api_key_here
export SENTIMENT_AGENT_PORT=10000  # Optional
```

### 3. Start the Sentiment Analysis Agent Server

Run the agent server using one of the launcher scripts:

```bash
cd ~/Desktop/A2A_with_MCP
python -m sentiment_analysis_agent.run_agent_server
```

Or alternatively:

```bash
cd ~/Desktop/A2A_with_MCP
python -m sentiment_analysis_agent.start_agent_server
```

The agent server will start on port 10000 by default. You can change this by setting the `SENTIMENT_AGENT_PORT` environment variable either in your .env file or directly in your environment.

### 4. Connect to the Agent

#### Using the A2A UI

1. Start the UI application
2. Go to the "Remote Agents" page
3. Add the agent with address `localhost:10000`
4. The UI will automatically detect the agent's capabilities via `/.well-known/agent.json`

#### Using the Orchestrator Agent

The agent can be used with the orchestrator agent, which will discover it through the standard A2A protocol endpoint at `/.well-known/agent.json`.

#### Direct API Interaction

You can interact with the agent directly using the A2A JSON-RPC protocol. Example request:

```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "method": "tasks/send",
  "params": {
    "id": "task-1",
    "sessionId": "session-1",
    "message": {
      "role": "user",
      "parts": [
        {
          "type": "text",
          "text": "What's the current sentiment about Bitcoin on Reddit?"
        }
      ]
    }
  }
}
```

## A2A Protocol Implementation

This agent implements the standard A2A protocol with the following endpoints:

- `/.well-known/agent.json` - Agent discovery metadata
- `/` - Main JSON-RPC endpoint for all A2A operations
- `/health` - Health check endpoint

Supported A2A methods:
- `tasks/send` - Send a task to the agent
- `tasks/get` - Get the current state of a task
- `tasks/cancel` - Cancel a running task

## API Configuration

### Base URL
- **Default URL**: `http://localhost:10000/`
- **Environment Variable**: `SENTIMENT_AGENT_PORT` (defaults to 10000)
- **Usage**: The base URL where the agent API is accessible

### Capabilities
- **Streaming**: `false` - This agent does not support streaming responses
- **Push Notifications**: `false` - This agent does not support push notifications
- **State Transition History**: `true` - This agent maintains task state history

### Supported JSON-RPC Methods

| Method | Description |
|--------|-------------|
| `tasks/send` | Send a sentiment analysis task to the agent |
| `tasks/get` | Get information about an existing task |
| `tasks/cancel` | Cancel a running task |

### Example Payloads

#### Example Request (tasks/send)

```json
{
  "jsonrpc": "2.0",
  "id": "request-123",
  "method": "tasks/send",
  "params": {
    "id": "task-abc123",
    "sessionId": "session-456",
    "message": {
      "role": "user",
      "parts": [
        {
          "type": "text",
          "text": "What's the current sentiment about Bitcoin on Reddit?"
        }
      ]
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
    "id": "task-abc123",
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
    "id": "task-abc123",
    "sessionId": "session-456",
    "status": {
      "state": "COMPLETED",
      "timestamp": "2023-06-14T18:31:30.456Z",
      "message": {
        "role": "assistant",
        "parts": [
          {
            "type": "text",
            "text": "# Bitcoin Sentiment Analysis (from r/Bitcoin)\n\n## Overall Sentiment: Positive\n\n- **Sentiment Score**: 0.65 (on a scale from -1 to 1)\n- **Positive Posts**: 12\n- **Negative Posts**: 3\n- **Neutral Posts**: 5\n- **Total Posts Analyzed**: 20\n\n## Key Topics:\n- Price movement and predictions\n- Institutional adoption\n- Regulatory news\n- Mining activity\n\n## Recent Discussion Topics:\n- \"Bitcoin breaks $60k resistance level\"\n- \"Major bank announces Bitcoin custody service\"\n- \"How mining difficulty changes affect the network\"\n\n## Summary:\nThe Bitcoin community on Reddit is currently showing a positive sentiment with a score of 0.65. Most discussions focus on the recent price increase above $60,000 and growing institutional adoption. There is optimism about sustained growth, though some cautionary posts about potential market volatility exist. Mining discussions are neutral and technical in nature."
          }
        ]
      }
    },
    "artifacts": [
      {
        "name": "sentiment-analysis",
        "description": "Bitcoin sentiment analysis results",
        "parts": [
          {
            "type": "text",
            "text": "# Bitcoin Sentiment Analysis (from r/Bitcoin)\n\n## Overall Sentiment: Positive\n\n- **Sentiment Score**: 0.65 (on a scale from -1 to 1)\n- **Positive Posts**: 12\n- **Negative Posts**: 3\n- **Neutral Posts**: 5\n- **Total Posts Analyzed**: 20\n\n## Key Topics:\n- Price movement and predictions\n- Institutional adoption\n- Regulatory news\n- Mining activity\n\n## Recent Discussion Topics:\n- \"Bitcoin breaks $60k resistance level\"\n- \"Major bank announces Bitcoin custody service\"\n- \"How mining difficulty changes affect the network\"\n\n## Summary:\nThe Bitcoin community on Reddit is currently showing a positive sentiment with a score of 0.65. Most discussions focus on the recent price increase above $60,000 and growing institutional adoption. There is optimism about sustained growth, though some cautionary posts about potential market volatility exist. Mining discussions are neutral and technical in nature."
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
            "text": "What's the current sentiment about Bitcoin on Reddit?"
          }
        ]
      },
      {
        "role": "assistant",
        "parts": [
          {
            "type": "text",
            "text": "# Bitcoin Sentiment Analysis (from r/Bitcoin)\n\n## Overall Sentiment: Positive\n\n- **Sentiment Score**: 0.65 (on a scale from -1 to 1)\n- **Positive Posts**: 12\n- **Negative Posts**: 3\n- **Neutral Posts**: 5\n- **Total Posts Analyzed**: 20\n\n## Key Topics:\n- Price movement and predictions\n- Institutional adoption\n- Regulatory news\n- Mining activity\n\n## Recent Discussion Topics:\n- \"Bitcoin breaks $60k resistance level\"\n- \"Major bank announces Bitcoin custody service\"\n- \"How mining difficulty changes affect the network\"\n\n## Summary:\nThe Bitcoin community on Reddit is currently showing a positive sentiment with a score of 0.65. Most discussions focus on the recent price increase above $60,000 and growing institutional adoption. There is optimism about sustained growth, though some cautionary posts about potential market volatility exist. Mining discussions are neutral and technical in nature."
          }
        ]
      }
    ]
  }
}
```

## Troubleshooting

If you encounter any issues:

1. **Check Logs**: Review the agent server logs in `sentiment_agent_server.log` and `sentiment_task_manager.log`

2. **API Key**: Verify your GOOGLE_API_KEY is correctly set in your environment or .env file

3. **A2A Compatibility**: Use tools like `curl` to verify the agent card is accessible:
   ```
   curl http://localhost:10000/.well-known/agent.json
   ```

## Architecture

This project consists of:

1. **Sentiment Analysis Agent**: A CrewAI agent that analyzes Bitcoin sentiment from Reddit
2. **A2A Task Manager**: Manages A2A task lifecycle and interfaces with the agent
3. **A2A Server**: Implements the A2A protocol endpoints
4. **MCP Server for Reddit**: A server that provides access to Reddit data

The flow is:
A2A Request → Server → Task Manager → Agent → MCP Server → Reddit API → Analysis → A2A Response 