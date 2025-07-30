#!/usr/bin/env python3
"""
Simple test to verify streaming is working
"""
import asyncio
import httpx
import json
from uuid import uuid4

async def simple_streaming_test():
    """Test basic streaming functionality"""
    
    print("🧪 Simple Streaming Test")
    print("="*40)
    
    async with httpx.AsyncClient() as client:
        
        # Check if Financial Agent supports streaming
        print("1. Checking agent capabilities...")
        agents_response = await client.post(
            "http://localhost:12000/agent/list",
            json={},
            timeout=5.0
        )
        
        agents = agents_response.json().get('result', [])
        financial_agent = next((a for a in agents if 'Financial' in a.get('name', '')), None)
        
        if financial_agent and financial_agent.get('capabilities', {}).get('streaming'):
            print("   ✅ Financial Agent supports streaming")
        else:
            print("   ❌ Streaming not supported or agent not found")
            return
        
        # Test direct streaming to Financial Agent
        print("\n2. Testing direct streaming to Financial Agent...")
        direct_request = {
            "id": str(uuid4()),
            "jsonrpc": "2.0",
            "method": "message/stream",
            "params": {
                "id": str(uuid4()),
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Give me Apple's P/E ratio"}]
                }
            }
        }
        
        try:
            response = await client.post(
                "http://localhost:8001/",
                json=direct_request,
                headers={"Accept": "text/event-stream"},
                timeout=10.0
            )
            
            print(f"   Direct streaming response: {response.status_code}")
            if response.status_code == 200:
                print("   ✅ Direct streaming works!")
                # Show first part of response
                response_text = response.text[:300]
                print(f"   Response preview: {response_text}...")
            else:
                print(f"   ❌ Direct streaming failed: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Direct streaming error: {e}")
        
        # Test via UI 
        print("\n3. Testing via UI...")
        
        # Create conversation
        conv_response = await client.post(
            "http://localhost:12000/conversation/create",
            json={},
            timeout=5.0
        )
        
        if conv_response.status_code == 200:
            conversation_id = conv_response.json()['result']['conversation_id']
            print(f"   ✅ Conversation created: {conversation_id[:8]}...")
            
            # Send message
            message = {
                "id": str(uuid4()),
                "role": "user",
                "parts": [{"type": "text", "text": "What is Apple's current P/E ratio?"}],
                "metadata": {
                    "conversation_id": conversation_id,
                    "message_id": str(uuid4())
                }
            }
            
            msg_response = await client.post(
                "http://localhost:12000/message/send",
                json={"params": message},
                timeout=15.0
            )
            
            if msg_response.status_code == 200:
                print("   ✅ Message sent via UI")
                
                # Quick check for immediate response
                await asyncio.sleep(2)
                
                messages_response = await client.post(
                    "http://localhost:12000/message/list",
                    json={"params": {"conversation_id": conversation_id}},
                    timeout=5.0
                )
                
                if messages_response.status_code == 200:
                    messages = messages_response.json().get('result', [])
                    print(f"   Found {len(messages)} messages total")
                    
                    agent_messages = [m for m in messages if m.get('role') == 'agent']
                    if agent_messages:
                        print("   ✅ Agent response received!")
                        response_text = agent_messages[0].get('parts', [{}])[0].get('text', 'No text')
                        print(f"   Response: {response_text[:100]}...")
                    else:
                        print("   ⚠️  No agent response yet (may still be processing)")
                
            else:
                print(f"   ❌ Failed to send message via UI: {msg_response.status_code}")
        else:
            print(f"   ❌ Failed to create conversation: {conv_response.status_code}")

    print("\n" + "="*40)
    print("🏁 Simple streaming test completed!")

if __name__ == "__main__":
    asyncio.run(simple_streaming_test())