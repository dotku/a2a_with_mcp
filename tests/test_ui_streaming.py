#!/usr/bin/env python3
"""
Test script to verify SSE streaming integration with Financial Analysis Agent
"""
import asyncio
import httpx
import json
from uuid import uuid4

async def test_streaming_integration():
    """Test streaming integration with polling for updates"""
    
    print("🧪 Testing SSE Streaming Integration")
    print("="*60)
    
    async with httpx.AsyncClient() as client:
        
        # Create conversation
        print("1. Creating conversation...")
        conversation_response = await client.post(
            "http://localhost:12000/conversation/create",
            json={},
            timeout=5.0
        )
        
        conversation_id = conversation_response.json()['result']['conversation_id']
        print(f"   ✅ Conversation: {conversation_id}")
        
        # Send message
        print("\n2. Sending financial query...")
        test_message = {
            "id": str(uuid4()),
            "role": "user", 
            "parts": [{"type": "text", "text": "What is Apple's current stock price and P/E ratio?"}],
            "metadata": {
                "conversation_id": conversation_id,
                "message_id": str(uuid4())
            }
        }
        
        message_response = await client.post(
            "http://localhost:12000/message/send",
            json={"params": test_message},
            timeout=30.0
        )
        
        if message_response.status_code == 200:
            print("   ✅ Message sent successfully")
        else:
            print(f"   ❌ Message failed: {message_response.status_code}")
            return
        
        # Poll for updates with longer wait times
        print("\n3. Polling for responses (this may take up to 2 minutes for financial analysis)...")
        max_polls = 24  # 2 minutes with 5-second intervals
        poll_count = 0
        
        while poll_count < max_polls:
            poll_count += 1
            await asyncio.sleep(5)
            
            print(f"   Poll {poll_count}/{max_polls} - Checking for updates...")
            
            # Check messages
            messages_response = await client.post(
                "http://localhost:12000/message/list",
                json={"params": {"conversation_id": conversation_id}},
                timeout=5.0
            )
            
            if messages_response.status_code == 200:
                messages = messages_response.json().get('result', [])
                agent_messages = [msg for msg in messages if msg.get('role') == 'agent']
                
                if agent_messages:
                    print(f"\n   ✅ Found {len(agent_messages)} agent response(s)!")
                    for i, msg in enumerate(agent_messages):
                        text = msg.get('parts', [{}])[0].get('text', 'No text')
                        print(f"   Response {i+1}: {text[:200]}...")
                    break
                else:
                    print(f"      No agent responses yet ({len(messages)} total messages)")
            
            # Check tasks
            tasks_response = await client.post(
                "http://localhost:12000/task/list",
                json={},
                timeout=5.0
            )
            
            if tasks_response.status_code == 200:
                tasks = tasks_response.json().get('result', [])
                if tasks:
                    print(f"      Found {len(tasks)} task(s):")
                    for task in tasks:
                        status = task.get('status', {})
                        state = status.get('state', 'unknown')
                        print(f"        Task {task.get('id', 'unknown')[:8]}: {state}")
                        
                        if state == 'completed' and status.get('message'):
                            task_text = status['message'].get('parts', [{}])[0].get('text', 'No text')
                            print(f"        Completed message: {task_text[:100]}...")
                            print("\n   ✅ Task completed! Streaming worked!")
                            return
                        elif state == 'working':
                            print("        Financial analysis in progress...")
                        elif state == 'failed':
                            print("        ❌ Task failed")
                            return
                else:
                    print("      No tasks found")
            
            # Check events
            events_response = await client.post(
                "http://localhost:12000/event/list",
                json={},
                timeout=5.0
            )
            
            if events_response.status_code == 200:
                events = events_response.json().get('result', [])
                recent_events = events[-3:]  # Show last 3 events
                if recent_events:
                    print(f"      Recent events ({len(events)} total):")
                    for event in recent_events:
                        actor = event.get('actor', 'unknown')
                        content = event.get('content', {})
                        if isinstance(content, dict) and content.get('parts'):
                            text = content['parts'][0].get('text', 'No text')[:50]
                            print(f"        {actor}: {text}...")
        
        print(f"\n   ⏰ Polling completed after {poll_count} attempts")
        
        # Final summary
        print("\n4. Final status summary...")
        
        # Final message check
        messages_response = await client.post(
            "http://localhost:12000/message/list",
            json={"params": {"conversation_id": conversation_id}},
            timeout=5.0
        )
        
        if messages_response.status_code == 200:
            messages = messages_response.json().get('result', [])
            print(f"   Total messages: {len(messages)}")
            
            for msg in messages:
                role = msg.get('role', 'unknown')
                text = msg.get('parts', [{}])[0].get('text', 'No text')[:100]
                print(f"   {role}: {text}...")
        
        # Final task check  
        tasks_response = await client.post(
            "http://localhost:12000/task/list",
            json={},
            timeout=5.0
        )
        
        if tasks_response.status_code == 200:
            tasks = tasks_response.json().get('result', [])
            print(f"   Total tasks: {len(tasks)}")
            
            for task in tasks:
                task_id = task.get('id', 'unknown')[:8]
                state = task.get('status', {}).get('state', 'unknown')
                print(f"   Task {task_id}: {state}")

    print("\n" + "="*60)
    print("🏁 Streaming integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_streaming_integration())