# MCP Integration Rules for Financial Agent

## Event Loop Management

- **Rule 1:** When working with MCP tools, always maintain a persistent event loop. Never create and close event loops for individual operations.
- **Rule 2:** Use a shared event loop for all MCP operations to prevent "Event loop is closed" errors.
- **Rule 3:** Handle event loop cleanup carefully at application exit - never assume the loop is active.

## Async MCP Tools

- **Rule 4:** Always use `.ainvoke()` (not `.invoke()`) with LangGraph apps that use MCP tools.
- **Rule 5:** Ensure tools returned by MCP clients are properly bound to the LLM with `.bind_tools()`.
- **Rule 6:** Keep tool initialization separate from tool usage - first initialize all tools, then bind them to models.

## LangGraph Configuration

- **Rule 7:** When setting up a LangGraph workflow with MCP tools, use a standard `ToolNode` not a custom executor.
- **Rule 8:** Set reasonable `max_iterations` values (10-20) to prevent excessive tool use.
- **Rule 9:** Include proper error handling in both the agent and tools nodes.

## Database Schema

- **Rule 10:** Ensure the LLM's system prompt contains accurate database schema information including table names, columns, and example queries.
- **Rule 11:** Include Foreign Key relationships in the schema to help the LLM understand data relationships.
- **Rule 12:** Provide example queries in system prompts that match the actual database schema.

## Error Handling and Logging

- **Rule 13:** Implement comprehensive error logging for both MCP client initialization and tool execution.
- **Rule 14:** Always provide fallback mechanisms for when MCP tools fail.
- **Rule 15:** Log all important state transitions and tool invocations for debugging.

## Testing

- **Rule 16:** Create dedicated test scripts that isolate MCP functionality from the LLM.
- **Rule 17:** Test database connections and queries directly before integrating with the agent.
- **Rule 18:** Use smaller, focused test cases that verify one aspect of functionality at a time.

## Application Lifecycle

- **Rule 19:** Properly clean up MCP resources during application shutdown.
- **Rule 20:** Implement lazy loading for MCP resources to prevent initialization issues.
- **Rule 21:** Use singleton patterns for MCP clients to prevent duplicated connections.

## Integration with External Systems

- **Rule 22:** Ensure database credentials are properly loaded from environment variables.
- **Rule 23:** Isolate MCP server configuration to make it easily modifiable.
- **Rule 24:** Verify MCP server processes are properly managed and monitored.

## Resource Usage

- **Rule 25:** Monitor event loop resource usage in production environments.
- **Rule 26:** Limit the complexity of database queries that can be run through MCP tools.
- **Rule 27:** Implement timeouts for long-running database operations. 