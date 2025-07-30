#!/usr/bin/env python3
"""
Test script to verify financial agent streaming response format
"""
import json
import asyncio
import httpx
from uuid import uuid4

async def test_financial_agent_streaming():
    """Test the financial agent's message streaming endpoint"""
    
    # Test payload - message/stream request
    test_request = {
        "id": str(uuid4()),
        "jsonrpc": "2.0",
        "method": "message/stream",
        "params": {
            "id": str(uuid4()),
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "What is the current P/E ratio for Apple?"
                    }
                ]
            }
        }
    }
    
    print(f"Testing financial agent at http://localhost:8001")
    print(f"Request: {json.dumps(test_request, indent=2)}")
    print("\n" + "="*50)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8001/",
                json=test_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream"
                },
                timeout=30.0
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            print(f"Response content: {response.text[:1000]}...")
            
            if response.status_code == 200:
                # Try to parse SSE events
                lines = response.text.split('\n')
                for line in lines:
                    if line.startswith('data: ') and not line.startswith('data: [DONE]'):
                        try:
                            data = line[6:]  # Remove 'data: ' prefix
                            parsed = json.loads(data)
                            print(f"\nParsed SSE event:")
                            print(json.dumps(parsed, indent=2))
                            
                            # Validate A2A format
                            if 'id' in parsed and 'jsonrpc' in parsed:
                                if parsed.get('jsonrpc') == '2.0':
                                    if 'result' in parsed:
                                        print("✅ Valid A2A streaming message response format")
                                    elif 'error' in parsed:
                                        print("✅ Valid A2A streaming error response format")
                                    else:
                                        print("❌ Missing 'result' or 'error' field")
                                else:
                                    print(f"❌ Invalid jsonrpc version: {parsed.get('jsonrpc')}")
                            else:
                                print("❌ Missing required A2A fields (id, jsonrpc)")
                        except json.JSONDecodeError as e:
                            print(f"❌ Failed to parse SSE data as JSON: {e}")
                            print(f"Raw data: {data}")
            else:
                print(f"❌ Request failed with status {response.status_code}")
                
        except httpx.ConnectError:
            print("❌ Could not connect to financial agent at http://localhost:8001")
            print("Make sure the financial agent is running")
        except Exception as e:
            print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    asyncio.run(test_financial_agent_streaming())