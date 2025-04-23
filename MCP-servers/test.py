import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def main():
    params = StdioServerParameters(
        command="python3",
        args=["postgres_mcp.py"],
    )

    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            print("✅ Connected to MCP")

            tools = await session.list_tools()
            print("HERE ARE THE TOOLS:",tools)
            print("🛠️  Tools:", [t[0] for t in tools])

            result = await session.call_tool("query", {"sql": "SELECT * FROM users;"})
            print("📦 Query result:", result)

if __name__ == "__main__":
    asyncio.run(main())
