from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
import re


# --8<-- [start:HelloWorldAgent]
class HelloWorldAgent:
    """Hello World Agent."""

    async def invoke(self, message: str = "") -> str:
        # Check if the message is about currency conversion
        if "USD" in message.upper() and "INR" in message.upper():
            # Extract numbers from the message
            numbers = re.findall(r'\d+', message)
            if numbers:
                usd_amount = int(numbers[0])
                # Approximate conversion rate (in real implementation, use live API)
                inr_amount = usd_amount * 83  # Approximate USD to INR rate
                return f"{usd_amount} USD is approximately {inr_amount} INR (using approximate rate of 1 USD = 83 INR)"
        
        # Default greeting for other messages
        return "hello world"


# --8<-- [end:HelloWorldAgent]


# --8<-- [start:HelloWorldAgentExecutor_init]
class HelloWorldAgentExecutor(AgentExecutor):
    """Test AgentProxy Implementation."""

    def __init__(self):
        self.agent = HelloWorldAgent()

    # --8<-- [end:HelloWorldAgentExecutor_init]
    # --8<-- [start:HelloWorldAgentExecutor_execute]
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # Extract the user message from the context
        user_message = ""
        # if context.request and hasattr(context.request, 'params') and context.request.params:
        #     if hasattr(context.request.params, 'message') and context.request.params.message:
        #         if hasattr(context.request.params.message, 'parts') and context.request.params.message.parts:
        #             for part in context.request.params.message.parts:
        #                 if hasattr(part, 'text'):
        #                     user_message += part.text + " "
        
        user_message = context.get_user_input()
        
        result = await self.agent.invoke(user_message)
        
        # result = await self.agent.invoke(user_message.strip())
        
        await event_queue.enqueue_event(new_agent_text_message(result))

    # --8<-- [end:HelloWorldAgentExecutor_execute]

    # --8<-- [start:HelloWorldAgentExecutor_cancel]
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception('cancel not supported')

    # --8<-- [end:HelloWorldAgentExecutor_cancel]
