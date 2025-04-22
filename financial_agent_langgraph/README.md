# Financial Analysis Agent

This agent provides financial analysis capabilities for the Multi-Agent Market Research & Financial Analysis System. It's built using LangGraph and follows the Agent-to-Agent (A2A) protocol for communication.

## Features

- Analyze financial metrics (P/E ratio, profit margins, etc.)
- Provide trend analysis for financial data
- Calculate valuation metrics and investment ratings
- Track market indices and trends

## Current Implementation

This version uses dummy data instead of connecting to a PostgreSQL MCP server. The agent is set up with:

- A LangGraph-based workflow for financial analysis
- Tools to fetch and analyze company financial metrics
- A FastAPI server compatible with the A2A protocol
- Support for streaming responses and push notifications

## Running the Agent

1. Ensure you have Python 3.10+ installed
2. Install dependencies with pip:
   ```
   pip install -r requirements.txt
   ```
3. Run the server:
   ```
   python -m financial_agent_langgraph
   ```
   
The server will start on port 8001 by default. You can change this in the `.env` file.

## API Endpoints

- `POST /`: Main JSON-RPC endpoint for A2A requests
- `GET /.well-known/ai-plugin.json`: Returns the agent card in AI plugin format
- `GET /health`: Health check endpoint
- `WebSocket /ws`: WebSocket endpoint for streaming responses

## A2A Methods Supported

- `tasks/send`: Send a task for financial analysis
- `tasks/sendSubscribe`: Send a task and subscribe to streaming updates
- `tasks/get`: Get a task by ID
- `tasks/cancel`: Cancel a task by ID
- `tasks/pushNotification/set`: Set push notification configuration for a task
- `tasks/pushNotification/get`: Get push notification configuration for a task
- `tasks/resubscribe`: Resubscribe to task updates

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