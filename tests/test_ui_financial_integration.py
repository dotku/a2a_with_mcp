#!/usr/bin/env python3
"""
Test script to verify end-to-end integration of Financial Analysis Agent with demo UI
"""
import asyncio
import httpx
import json
from uuid import uuid4

async def test_ui_financial_integration():
    """Test the complete flow: UI -> Financial Agent -> SSE streaming"""
    
    print("🧪 Testing Financial Analysis Agent integration with demo UI")
    print("="*60)
    
    async with httpx.AsyncClient() as client:
        
        # Step 1: Verify agent is registered
        print("1. Checking registered agents...")
        agents_response = await client.post(
            "http://localhost:12000/agent/list",
            json={},
            timeout=5.0
        )
        
        if agents_response.status_code == 200:
            agents = agents_response.json()
            financial_agent = None
            for agent in agents.get('result', []):
                if 'Financial Analysis' in agent.get('name', ''):
                    financial_agent = agent
                    break
            
            if financial_agent:
                print(f"   ✅ Financial Analysis Agent found: {financial_agent['name']}")
                print(f"   ✅ Streaming supported: {financial_agent['capabilities']['streaming']}")
            else:
                print("   ❌ Financial Analysis Agent not found in registered agents")
                return
        else:
            print(f"   ❌ Failed to list agents: {agents_response.status_code}")
            return
        
        # Step 2: Create a conversation 
        print("\n2. Creating conversation...")
        conversation_response = await client.post(
            "http://localhost:12000/conversation/create",
            json={},
            timeout=5.0
        )
        
        if conversation_response.status_code == 200:
            conversation = conversation_response.json()
            conversation_id = conversation['result']['conversation_id']
            print(f"   ✅ Conversation created: {conversation_id}")
        else:
            print(f"   ❌ Failed to create conversation: {conversation_response.status_code}")
            return
        
        # Step 3: Send a financial query message
        print("\n3. Sending financial query message...")
        test_message = {
            "id": str(uuid4()),
            "role": "user", 
            "parts": [
                {
                    "type": "text",
                    "text": "What is the current P/E ratio for Apple stock?"
                }
            ],
            "metadata": {
                "conversation_id": conversation_id,
                "message_id": str(uuid4())
            }
        }
        
        message_response = await client.post(
            "http://localhost:12000/message/send",
            json={"params": test_message},
            timeout=30.0  # Financial analysis can take time
        )
        
        print(f"   Message response status: {message_response.status_code}")
        print(f"   Message response: {message_response.text[:500]}...")
        
        if message_response.status_code == 200:
            print("   ✅ Message sent successfully")
        else:
            print("   ❌ Failed to send message")
            return
        
        # Step 4: Check for response by listing messages
        print("\n4. Checking for agent response...")
        await asyncio.sleep(3)  # Give time for processing
        
        messages_response = await client.post(
            "http://localhost:12000/message/list",
            json={"params": {"conversation_id": conversation_id}},
            timeout=5.0
        )
        
        if messages_response.status_code == 200:
            messages = messages_response.json()
            print(f"   ✅ Retrieved {len(messages.get('result', []))} messages")
            
            # Look for agent responses
            agent_responses = [msg for msg in messages.get('result', []) if msg.get('role') == 'agent']
            if agent_responses:
                print(f"   ✅ Found {len(agent_responses)} agent response(s)")
                for i, response in enumerate(agent_responses):
                    print(f"   Response {i+1}: {response.get('parts', [{}])[0].get('text', 'No text')[:100]}...")
            else:
                print("   ⚠️  No agent responses found yet")
        else:
            print(f"   ❌ Failed to list messages: {messages_response.status_code}")
        
        # Step 5: Check tasks (if any were created)
        print("\n5. Checking tasks...")
        tasks_response = await client.post(
            "http://localhost:12000/task/list", 
            json={},
            timeout=5.0
        )
        
        if tasks_response.status_code == 200:
            tasks = tasks_response.json()
            task_list = tasks.get('result', [])
            print(f"   ✅ Found {len(task_list)} task(s)")
            
            for i, task in enumerate(task_list):
                print(f"   Task {i+1}: {task.get('id', 'Unknown ID')} - Status: {task.get('status', {}).get('state', 'Unknown state')}")
                if task.get('status', {}).get('message', {}).get('parts'):
                    task_message = task['status']['message']['parts'][0].get('text', 'No text')
                    print(f"            Message: {task_message[:100]}...")
        else:
            print(f"   ❌ Failed to list tasks: {tasks_response.status_code}")
    
    print("\n" + "="*60)
    print("🏁 Integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_ui_financial_integration())