#!/usr/bin/env python3
"""
Script to register the Financial Analysis Agent with the demo UI
"""
import asyncio
import httpx

async def register_financial_agent():
    """Register the Financial Analysis Agent with the demo UI"""
    
    # First, check if the Financial Agent is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8001/.well-known/agent.json")
            if response.status_code == 200:
                agent_info = response.json()
                print(f"✅ Financial Agent is running: {agent_info.get('name', 'Unknown')}")
            else:
                print(f"❌ Financial Agent health check failed: {response.status_code}")
                return
    except Exception as e:
        print(f"❌ Could not connect to Financial Agent at http://localhost:8001: {e}")
        return
    
    # Register with the demo UI
    try:
        async with httpx.AsyncClient() as client:
            register_request = {
                "params": "localhost:8001"
            }
            
            print(f"Registering agent with request: {register_request}")
            
            response = await client.post(
                "http://localhost:12000/agent/register",
                json=register_request,
                timeout=10.0
            )
            
            print(f"Registration response status: {response.status_code}")
            print(f"Registration response: {response.text}")
            
            if response.status_code == 200:
                print("✅ Financial Analysis Agent registered successfully with demo UI")
                
                # Verify registration by listing agents
                list_response = await client.post(
                    "http://localhost:12000/agent/list",
                    json={},
                    timeout=5.0
                )
                if list_response.status_code == 200:
                    agents = list_response.json()
                    print(f"Registered agents: {agents}")
                
            else:
                print(f"❌ Failed to register agent: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Error registering agent with demo UI: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure the demo UI server is running on http://localhost:12000")

if __name__ == "__main__":
    asyncio.run(register_financial_agent())