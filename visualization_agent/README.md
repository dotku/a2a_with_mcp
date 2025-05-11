# Data Visualization Agent

This agent generates visualizations and charts from data based on descriptions. It's part of the Multi-Agent Market Research & Financial Analysis System, implementing the Agent-to-Agent (A2A) protocol for interoperability with other agents and orchestrators.

## Features

- Generates various chart types (bar charts, line charts, pie charts)
- Accepts data in JSON format
- Returns visualizations as PNG images
- Implements A2A protocol for agent discovery and task management
- Uses LRU caching for efficient storage of visualizations

## Architecture

- **Core Engine**: Built using Matplotlib for visualization generation
- **Caching**: LRU (Least Recently Used) caching mechanism for storing visualizations
- **A2A Protocol**: Implements standard A2A endpoints and methods
- **Error Handling**: Robust error handling with consistent JSON-RPC error responses

## API Configuration

### Base URL
- **Default URL**: `http://localhost:8004/`
- **Usage**: The base URL where the visualization agent API is accessible

### Capabilities
- **Streaming**: `false` - This agent does not support streaming responses
- **Push Notifications**: `false` - This agent does not support push notifications
- **State Transition History**: `false` - This agent does not maintain task state history

## API Endpoints

- `POST /`: Main JSON-RPC endpoint for A2A requests
- `GET /.well-known/agent.json`: Returns the agent card/manifest
- `GET /health`: Health check endpoint

## Supported JSON-RPC Methods

| Method | Description |
|--------|-------------|
| `tasks/send` | Send a visualization task to the agent |
| `tasks/get` | Get information about an existing task |
| `tasks/cancel` | Cancel a running task |

## Example Payloads

### Example Request (tasks/send)

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
          "text": "{\"plot_description\": \"A bar chart showing monthly revenue\", \"data_json\": \"{\\\"labels\\\": [\\\"Jan\\\", \\\"Feb\\\", \\\"Mar\\\", \\\"Apr\\\", \\\"May\\\"], \\\"values\\\": [10000, 12000, 9000, 15000, 16000], \\\"title\\\": \\\"Monthly Revenue\\\", \\\"xlabel\\\": \\\"Month\\\", \\\"ylabel\\\": \\\"Revenue ($)\\\"}\"}"
        }
      ]
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
    "id": "task-abc123",
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
    "id": "task-abc123",
    "sessionId": "session-456",
    "status": {
      "state": "COMPLETED",
      "timestamp": "2023-06-14T18:31:30.456Z"
    },
    "artifacts": [
      {
        "name": "plot_abcdef1234.png",
        "description": "Visualization: A bar chart showing monthly revenue",
        "parts": [
          {
            "file": {
              "mimeType": "image/png",
              "name": "plot_abcdef1234.png",
              "bytes": "base64_encoded_image_data_here..."
            }
          }
        ]
      }
    ]
  }
}
```

## Error Handling

The visualization agent uses standard JSON-RPC error codes:

- `-32600`: Invalid Request - The JSON sent is not a valid Request object
- `-32601`: Method not found - The method does not exist or is not available
- `-32602`: Invalid params - Invalid method parameter(s)
- `-32603`: Internal error - Internal JSON-RPC error
- `-32000`: Server error - Visualization generation error

Example error response:

```json
{
  "jsonrpc": "2.0",
  "id": "request-123",
  "error": {
    "code": -32603,
    "message": "Error generating visualization: Invalid data format"
  }
}
```

## Setup & Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the server:
   ```bash
   python -m visualization_agent
   ```

3. The server will start on port 8004 by default.

## Cache Management

Visualizations are stored in an LRU (Least Recently Used) cache with a default capacity of 100 items. When the cache reaches its limit, the least recently accessed visualizations will be automatically removed. 