import asyncio
import logging
import json
from typing import Any
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)


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
            logger.info(agent_card.model_dump_json(indent=2, exclude_none=True))
        except Exception as e:
            logger.error(f'Failed to fetch agent card: {e}')
            return

        # Initialize A2A client
        client = A2AClient(
            httpx_client=httpx_client, 
            agent_card=agent_card
        )
        logger.info('A2AClient initialized for orchestrator.')

        # Example travel planning request
        travel_request = "Plan a 3-day 5 persons business econ class trip to Paris with a budget of $2000 from SFO on 2025-07-10 with return date of 2025-07-13. I need flights. Answer with best suggestions?"
        
        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': travel_request}
                ],
                'messageId': uuid4().hex,
            },
        }
        
        request = SendMessageRequest(
            id=str(uuid4()), 
            params=MessageSendParams(**send_message_payload)
        )

        logger.info(f'Sending travel request: {travel_request}')
        
        try:
            response = await client.send_message(request)
            logger.info('Response received:')
            response_dict = response.model_dump(mode='json', exclude_none=True)
            print(json.dumps(response_dict, indent=2))
        except Exception as e:
            logger.error(f'Error sending message: {e}')


if __name__ == '__main__':
    asyncio.run(main())
