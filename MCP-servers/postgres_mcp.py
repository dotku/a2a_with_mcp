import asyncio
import asyncpg
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server

# Global pool variable
default_pool: asyncpg.Pool | None = None

@asynccontextmanager
async def lifespan(server: FastMCP):
    global default_pool
    db_url = "postgresql://postgres:password@localhost:5432/testdb"
    default_pool = await asyncpg.create_pool(dsn=db_url)
    try:
        yield
    finally:
        await default_pool.close()

# Instantiate the server
mcp = FastMCP("Postgres MCP Server", lifespan=lifespan)

# Tool: Run read-only queries
@mcp.tool()
async def query(sql: str) -> list[dict]:
    assert default_pool is not None
    rows = await default_pool.fetch(sql)
    return [dict(row) for row in rows]

# Resource: View table schemas
@mcp.resource("schema://{schema_name}")
async def get_schema(schema_name: str) -> list[dict]:
    assert default_pool is not None
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
    assert default_pool is not None
    row = await default_pool.fetchrow(
        "SELECT data FROM financial_snapshot ORDER BY created_at DESC LIMIT 1"
    )
    return row["data"]  # already a Python dict thanks to asyncpg/jsonb

# Run the server on stdio


if __name__ == "__main__":
    mcp.run(transport="stdio")