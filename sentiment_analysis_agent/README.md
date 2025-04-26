# Bitcoin Sentiment Analysis Agent

This agent analyzes sentiment for Bitcoin discussions on Reddit using the CrewAI framework.

## Setup and Running

### 1. Install Dependencies

Make sure you have all required dependencies installed:

```bash
pip install crewai fastapi uvicorn httpx
```

### 2. Set Environment Variables

Set your Google API key:

```bash
export GOOGLE_API_KEY=your_api_key_here
```

### 3. Start the Sentiment Analysis Agent Server

Run the agent server using the launcher script that handles import paths correctly:

```bash
cd ~/Desktop/A2A_with_MCP
python -m sentiment_analysis_agent.run_agent_server
```

The agent server will start on port 10000 by default. You can change this by setting the `SENTIMENT_AGENT_PORT` environment variable.

### 4. Start the UI

Run the UI application:

```bash
cd ~/Desktop/A2A_with_MCP/demo/ui
uv run main.py
```

The UI server will start on port 12000 by default.

### 5. Register the Agent in the UI

1. Open your browser and navigate to `http://localhost:12000`
2. Go to the "Remote Agents" page
3. Click the "+" button to add a new agent
4. Enter `localhost:10000` as the agent address
5. Click "Read" to read the agent information
6. Click "Save" to register the agent

### 6. Use the Agent

1. Return to the main page
2. Start a new conversation
3. Select the "Bitcoin Sentiment Analyst" agent
4. Ask questions like "What's the current sentiment about Bitcoin on Reddit?"

## Troubleshooting

If you encounter any issues:

1. **Import Errors**: The project uses a directory named `crewai` which may conflict with the installed `crewai` package. Always use the `run_agent_server.py` script to start the server to avoid import conflicts.

2. **Check Logs**: Review the agent server logs in `sentiment_agent_server.log`

3. **API Key**: Verify your GOOGLE_API_KEY is correctly set and valid

4. **Connectivity**: Ensure network connectivity between the UI and agent servers

## Architecture

This project consists of:

1. **Sentiment Analysis Agent**: A CrewAI agent that analyzes Bitcoin sentiment from Reddit
2. **MCP Server for Reddit**: A server that provides access to Reddit data
3. **UI Integration**: Integration with the A2A UI framework for easy interaction

The flow is:
User → UI → Agent Server → MCP Server → Reddit API → Analysis → Response 