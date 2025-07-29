import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)


async def send_message_to_orchestrator(client: A2AClient, message: str, context_id: str = None, task_id: str = None) -> dict:
    """Send a message to the orchestrator and return the response."""
    send_message_payload: dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': [
                {'kind': 'text', 'text': message}
            ],
            'messageId': uuid4().hex,
        },
    }
    
    if context_id:
        send_message_payload['contextId'] = context_id
    if task_id:
        send_message_payload['taskId'] = task_id
    
    request = SendMessageRequest(
        id=str(uuid4()), 
        params=MessageSendParams(**send_message_payload)
    )

    response = await client.send_message(request)
    return response.model_dump(mode='json', exclude_none=True)


async def main() -> None:
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Orchestrator agent runs on port 10101
    base_url = 'http://localhost:10101'

    async with httpx.AsyncClient() as httpx_client:
        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        # Fetch the orchestrator agent card
        try:
            logger.info(f'Fetching orchestrator agent card from: {base_url}')
            agent_card = await resolver.get_agent_card()
            logger.info('Successfully fetched orchestrator agent card')
        except Exception as e:
            logger.error(f'Failed to fetch agent card: {e}')
            return

        # Initialize A2A client
        client = A2AClient(
            httpx_client=httpx_client, 
            agent_card=agent_card
        )
        logger.info('A2AClient initialized for orchestrator.')

        # Start the conversation
        print("\n=== Interactive Orchestrator Client ===")
        print("Type 'quit' to exit\n")
        
        context_id = None
        task_id = None
        
        # Initial travel planning request
        initial_request = "Plan a 3-day trip to Paris with a budget of $2000. I need flights, hotel, and car rental."
        print(f"🤖 Sending: {initial_request}")
        
        try:
            response = await send_message_to_orchestrator(client, initial_request)
            
            # Extract context and task IDs for follow-up messages
            if 'result' in response:
                context_id = response['result'].get('contextId')
                task_id = response['result'].get('id')
            
            print(f"\n📨 Response:")
            print(json.dumps(response, indent=2))
            
            # Check if orchestrator needs more input
            result = response.get('result', {})
            status = result.get('status', {})
            
            if status.get('state') == 'input-required':
                # Extract the question from the latest message
                history = result.get('history', [])
                if history:
                    latest_message = history[-1]
                    question_text = latest_message.get('parts', [{}])[0].get('text', '')
                    
                    # Try to parse JSON if it's structured
                    try:
                        if question_text.startswith('```json') and question_text.endswith('```'):
                            json_content = question_text.strip('```json\n').strip('```')
                            parsed = json.loads(json_content)
                            question = parsed.get('question', question_text)
                        else:
                            question = question_text
                    except:
                        question = question_text
                    
                    print(f"\n❓ Orchestrator asks: {question}")
                    
                    # Interactive follow-up
                    while True:
                        user_input = input("\n👤 Your answer (or 'quit' to exit): ").strip()
                        
                        if user_input.lower() == 'quit':
                            break
                            
                        if user_input:
                            print(f"\n🤖 Sending: {user_input}")
                            follow_up_response = await send_message_to_orchestrator(
                                client, user_input, context_id, task_id
                            )
                            
                            print(f"\n📨 Response:")
                            print(json.dumps(follow_up_response, indent=2))
                            
                            # Check if we need more input or if we're done
                            follow_up_result = follow_up_response.get('result', {})
                            follow_up_status = follow_up_result.get('status', {})
                            
                            if follow_up_status.get('state') != 'input-required':
                                print("\n✅ Travel planning complete!")
                                break
                            else:
                                # Extract next question
                                history = follow_up_result.get('history', [])
                                if history:
                                    latest_message = history[-1]
                                    question_text = latest_message.get('parts', [{}])[0].get('text', '')
                                    
                                    try:
                                        if question_text.startswith('```json') and question_text.endswith('```'):
                                            json_content = question_text.strip('```json\n').strip('```')
                                            parsed = json.loads(json_content)
                                            question = parsed.get('question', question_text)
                                        else:
                                            question = question_text
                                    except:
                                        question = question_text
                                    
                                    print(f"\n❓ Orchestrator asks: {question}")
            
        except Exception as e:
            logger.error(f'Error in conversation: {e}')


if __name__ == '__main__':
    asyncio.run(main())
