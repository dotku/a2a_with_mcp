import asyncio
import asyncpg
from contextlib import asynccontextmanager
import os # Import os for environment variables
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server

load_dotenv()

# Global pool variable
default_pool: asyncpg.Pool | None = None

@asynccontextmanager
async def lifespan(server: FastMCP):
    global default_pool
    
    # Construct db_url from environment variables with fallbacks
    # Replace 'strong_password_here' with the actual password you set for financial_agent_user
    pg_user = os.environ.get("PG_USER", "financial_agent_user")
    pg_password = os.environ.get("PG_PASSWORD", "strong_password_here") # IMPORTANT: Use your actual password here as fallback or ensure PG_PASSWORD is set
    pg_host = os.environ.get("PG_HOST", "localhost")
    pg_port = os.environ.get("PG_PORT", "5432")
    pg_database = os.environ.get("PG_DATABASE", "financial_agent_db")

    db_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
    
    print(f"Connecting to PostgreSQL with URL: {db_url.replace(pg_password, '********')}") # Log URL safely

    default_pool = await asyncpg.create_pool(dsn=db_url)
    try:
        yield
    finally:
        if default_pool: # Ensure pool exists before trying to close
            await default_pool.close()
            print("PostgreSQL connection pool closed.")

# Instantiate the server
mcp = FastMCP("Postgres MCP Server", lifespan=lifespan)

# Tool: Run read-only queries
@mcp.tool()
async def query(sql: str) -> list[dict]:
    assert default_pool is not None, "Database pool not initialized"
    rows = await default_pool.fetch(sql)
    return [dict(row) for row in rows]

# Resource: View table schemas
@mcp.resource("schema://{schema_name}")
async def get_schema(schema_name: str) -> list[dict]:
    assert default_pool is not None, "Database pool not initialized"
    rows = await default_pool.fetch(
        """
        SELECT table_name,
               json_agg(json_build_object(
                 'column', column_name,
                 'type', data_type
               ) ORDER BY ordinal_position) AS columns
        FROM information_schema.columns
        WHERE table_schema = $1
        GROUP BY table_name
        """,
        schema_name
    )
    return [{"table": r["table_name"], "columns": r["columns"]} for r in rows]

@mcp.tool()
async def fetch_financial_snapshot() -> dict:
    """
    Returns the most recent financial_snapshot.data JSON as a dict.
    """
    assert default_pool is not None, "Database pool not initialized"
    row = await default_pool.fetchrow(
        "SELECT data FROM financial_snapshot ORDER BY created_at DESC LIMIT 1"
    )
    if row and row["data"]:
        return row["data"]  # already a Python dict thanks to asyncpg/jsonb
    return {} # Return empty dict if no snapshot found or data is null

# Run the server on stdio
if __name__ == "__main__":
    mcp.run(transport="stdio")