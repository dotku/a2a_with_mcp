#!/usr/bin/env python3
"""
Debug script to trace exactly where the UI flow is failing
"""
import asyncio
import httpx
import json
import time
from uuid import uuid4

async def debug_ui_message_flow():
    """Debug the complete message flow with detailed logging"""
    
    print("🔍 Debugging UI Message Flow")
    print("="*50)
    
    async with httpx.AsyncClient() as client:
        
        # Step 1: Verify setup
        print("1. Verifying setup...")
        
        # Check UI is running
        try:
            ui_response = await client.post("http://localhost:12000/agent/list", json={}, timeout=5.0)
            if ui_response.status_code == 200:
                agents = ui_response.json().get('result', [])
                print(f"   ✅ UI running with {len(agents)} agents")
                
                financial_agent = next((a for a in agents if 'Financial' in a.get('name', '')), None)
                if financial_agent:
                    print(f"   ✅ Financial Agent registered: {financial_agent['name']}")
                    print(f"   ✅ Streaming: {financial_agent['capabilities']['streaming']}")
                else:
                    print("   ❌ Financial Agent not found - registering...")
                    register_response = await client.post(
                        "http://localhost:12000/agent/register",
                        json={"params": "localhost:8001"},
                        timeout=10.0
                    )
                    if register_response.status_code == 200:
                        print("   ✅ Agent registered successfully")
                    else:
                        print(f"   ❌ Registration failed: {register_response.status_code}")
                        return
            else:
                print(f"   ❌ UI not responding: {ui_response.status_code}")
                return
        except Exception as e:
            print(f"   ❌ UI connection error: {e}")
            return
        
        # Check Financial Agent is running
        try:
            fa_response = await client.get("http://localhost:8001/.well-known/agent.json", timeout=5.0)
            if fa_response.status_code == 200:
                print("   ✅ Financial Agent responding")
            else:
                print(f"   ❌ Financial Agent not responding: {fa_response.status_code}")
                return
        except Exception as e:
            print(f"   ❌ Financial Agent connection error: {e}")
            return
        
        # Step 2: Create conversation and send message
        print("\n2. Creating conversation and sending message...")
        
        # Create conversation
        conv_response = await client.post("http://localhost:12000/conversation/create", json={}, timeout=5.0)
        if conv_response.status_code != 200:
            print(f"   ❌ Failed to create conversation: {conv_response.status_code}")
            return
        
        conversation_id = conv_response.json()['result']['conversation_id']
        print(f"   ✅ Conversation created: {conversation_id}")
        
        # Send message
        message = {
            "id": str(uuid4()),
            "role": "user",
            "parts": [{"type": "text", "text": "What is Apple's P/E ratio?"}],
            "metadata": {
                "conversation_id": conversation_id,
                "message_id": str(uuid4())
            }
        }
        
        print(f"   📤 Sending message: {message['parts'][0]['text']}")
        
        msg_response = await client.post(
            "http://localhost:12000/message/send",
            json={"params": message},
            timeout=30.0
        )
        
        print(f"   📥 Message response status: {msg_response.status_code}")
        if msg_response.status_code == 200:
            response_data = msg_response.json()
            print(f"   📥 Message response: {json.dumps(response_data, indent=2)}")
        else:
            print(f"   ❌ Message failed: {msg_response.text}")
            return
        
        # Step 3: Monitor the processing with detailed polling
        print("\n3. Monitoring processing (60 second window)...")
        
        start_time = time.time()
        max_wait = 60  # 1 minute
        poll_interval = 3  # 3 seconds
        
        last_message_count = 0
        last_task_count = 0
        last_event_count = 0
        
        while (time.time() - start_time) < max_wait:
            elapsed = int(time.time() - start_time)
            print(f"\n   ⏱️  [{elapsed}s] Checking status...")
            
            # Check messages
            try:
                messages_response = await client.post(
                    "http://localhost:12000/message/list",
                    json={"params": {"conversation_id": conversation_id}},
                    timeout=5.0
                )
                
                if messages_response.status_code == 200:
                    messages = messages_response.json().get('result', [])
                    if len(messages) != last_message_count:
                        print(f"      📨 Messages: {len(messages)} (was {last_message_count})")
                        for i, msg in enumerate(messages):
                            role = msg.get('role', 'unknown')
                            text = msg.get('parts', [{}])[0].get('text', 'No text')[:60]
                            print(f"        {i+1}. {role}: {text}...")
                        last_message_count = len(messages)
                        
                        # If we got an agent response, check if it's the error message
                        agent_messages = [m for m in messages if m.get('role') == 'agent']
                        if agent_messages:
                            for agent_msg in agent_messages:
                                text = agent_msg.get('parts', [{}])[0].get('text', '')
                                if "I'm sorry, I encountered an error" in text:
                                    print(f"      ❌ Found error message: {text}")
                                    print("      🔍 This means the host runner failed to process the message")
                                    break
                                else:
                                    print(f"      ✅ Got valid agent response: {text[:100]}...")
                                    print("      🎉 Flow is working!")
                                    return
                    else:
                        print(f"      📨 Messages: {len(messages)} (no change)")
                else:
                    print(f"      ❌ Failed to get messages: {messages_response.status_code}")
            except Exception as e:
                print(f"      ❌ Message check error: {e}")
            
            # Check tasks
            try:
                tasks_response = await client.post("http://localhost:12000/task/list", json={}, timeout=5.0)
                if tasks_response.status_code == 200:
                    tasks = tasks_response.json().get('result', [])
                    if len(tasks) != last_task_count:
                        print(f"      🔧 Tasks: {len(tasks)} (was {last_task_count})")
                        for task in tasks:
                            task_id = task.get('id', 'unknown')[:8]
                            state = task.get('status', {}).get('state', 'unknown')
                            print(f"        Task {task_id}: {state}")
                        last_task_count = len(tasks)
                    else:
                        print(f"      🔧 Tasks: {len(tasks)} (no change)")
                else:
                    print(f"      ❌ Failed to get tasks: {tasks_response.status_code}")
            except Exception as e:
                print(f"      ❌ Task check error: {e}")
            
            # Check events
            try:
                events_response = await client.post("http://localhost:12000/event/list", json={}, timeout=5.0)
                if events_response.status_code == 200:
                    events = events_response.json().get('result', [])
                    if len(events) != last_event_count:
                        print(f"      📋 Events: {len(events)} (was {last_event_count})")
                        # Show last few events
                        recent_events = events[-3:]
                        for event in recent_events:
                            actor = event.get('actor', 'unknown')
                            content = event.get('content', {})
                            if isinstance(content, dict) and content.get('parts'):
                                text = content['parts'][0].get('text', 'No text')[:40]
                                print(f"        {actor}: {text}...")
                        last_event_count = len(events)
                    else:
                        print(f"      📋 Events: {len(events)} (no change)")
                else:
                    print(f"      ❌ Failed to get events: {events_response.status_code}")
            except Exception as e:
                print(f"      ❌ Event check error: {e}")
            
            await asyncio.sleep(poll_interval)
        
        print(f"\n   ⏰ Monitoring completed after {max_wait} seconds")
        
        # Step 4: Final status
        print("\n4. Final status summary...")
        
        # Get final messages
        try:
            messages_response = await client.post(
                "http://localhost:12000/message/list",
                json={"params": {"conversation_id": conversation_id}},
                timeout=5.0
            )
            
            if messages_response.status_code == 200:
                messages = messages_response.json().get('result', [])
                print(f"   📨 Final message count: {len(messages)}")
                
                agent_messages = [m for m in messages if m.get('role') == 'agent']
                if agent_messages:
                    for i, msg in enumerate(agent_messages):
                        text = msg.get('parts', [{}])[0].get('text', 'No text')
                        print(f"   Agent response {i+1}: {text}")
                        
                        if "I'm sorry, I encountered an error" in text:
                            print("   🔍 ERROR ANALYSIS:")
                            print("     - The ADKHostManager.process_message() method failed")
                            print("     - The _host_runner.run_async() either threw an exception or yielded no events")
                            print("     - This could be due to:")
                            print("       1. Host agent not finding the Financial Analysis Agent")
                            print("       2. Communication error between UI and Financial Agent")
                            print("       3. Authentication or configuration issue")
                            print("       4. Timeout in the financial analysis processing")
                else:
                    print("   ❌ No agent responses found")
            else:
                print(f"   ❌ Failed to get final messages: {messages_response.status_code}")
        except Exception as e:
            print(f"   ❌ Final message check error: {e}")

    print("\n" + "="*50)
    print("🔍 Debug completed!")

if __name__ == "__main__":
    asyncio.run(debug_ui_message_flow())