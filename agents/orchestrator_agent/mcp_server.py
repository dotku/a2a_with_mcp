import logging
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create an MCP server
mcp = FastMCP("Orchestrator Agent")

@mcp.resource("tasks://{task_id}/status")
def get_task_status(task_id: str) -> str:
    """
    Get the status of a task.
    
    Args:
        task_id: The ID of the task to check
        
    Returns:
        The status of the task
    """
    logger.info(f"Getting status for task {task_id}")
    return f"Status for task {task_id}: In Progress"

@mcp.resource("templates://{template_type}")
def get_template(template_type: str) -> str:
    """
    Get a template for a specific type of analysis.
    
    Args:
        template_type: The type of template to retrieve
        
    Returns:
        The template content
    """
    templates = {
        "market_analysis": "# Market Analysis Template\n1. Industry Overview\n2. Market Size and Growth\n3. Key Trends\n4. Competitive Landscape\n5. Future Outlook",
        "investment_thesis": "# Investment Thesis Template\n1. Company Overview\n2. Financials\n3. Competitive Position\n4. Growth Drivers\n5. Risks and Mitigations\n6. Valuation and Recommendation"
    }
    return templates.get(template_type, f"Template not found for {template_type}")

@mcp.tool()
def create_task(query: str, task_type: str) -> str:
    """
    Create a new task for orchestration.
    
    Args:
        query: The user query to process
        task_type: The type of task to create
        
    Returns:
        The created task ID
    """
    logger.info(f"Creating {task_type} task for query: {query}")
    return f"Created task with ID: task_{hash(query) % 1000000}"

@mcp.tool()
def delegate_task(agent_type: str, query: str) -> str:
    """
    Delegate a task to a specialized agent.
    
    Args:
        agent_type: The type of agent to delegate to
        query: The query to process
        
    Returns:
        The result of the delegation
    """
    logger.info(f"Delegating task to {agent_type} agent: {query}")
    return f"Task delegated to {agent_type} agent successfully" 