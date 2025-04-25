# Financial Agent with PostgreSQL MCP Integration

A powerful financial analysis agent built with LangGraph that connects to PostgreSQL databases using the Model Context Protocol (MCP).

## Overview

This project implements a financial analysis agent that:

- Processes natural language queries about financial data
- Connects to PostgreSQL databases using MCP for data access
- Uses LangGraph for orchestrating AI conversation flows with tool use
- Returns formatted financial insights based on live database data

## Architecture

The system consists of:

1. **Financial Agent**: A LangGraph-based agent that processes financial queries using LLMs
2. **MCP Postgres Server**: A Model Context Protocol server that exposes PostgreSQL tools to the LLM
3. **Database Integration**: Connection to a PostgreSQL database with financial data

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