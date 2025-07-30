import mesop as me
import mesop.labs as mel

import asyncio
import uuid
import functools
import threading

from state.state import AppState, SettingsState, StateMessage
from state.host_agent_service import SendMessage, ListConversations, convert_message_to_state
from .chat_bubble import chat_bubble
from .form_render import is_form, render_form, form_sent
from common.types import Message, TextPart


@me.stateclass
class PageState:
    """Local Page State"""
    conversation_id: str = ""
    message_content: str = ""


def on_blur(e: me.InputBlurEvent):
    """input handler"""
    state = me.state(PageState)
    state.message_content = e.value


async def send_message(message: str, message_id: str = ""):
  state = me.state(PageState)
  app_state = me.state(AppState)
  settings_state = me.state(SettingsState)
  c = next(
      (
          x
          for x in await ListConversations()
          if x.conversation_id == state.conversation_id
      ),
      None,
  )
  if not c:
    print("Conversation id ", state.conversation_id, " not found")
  request = Message(
      id=message_id,
      role="user",
      parts=[TextPart(text=message)],
      metadata={'conversation_id': c.conversation_id if c else "",
                'conversation_name': c.name if c else ""},
  )
  # Add message to state until refresh replaces it.
  state_message = convert_message_to_state(request)
  if not app_state.messages:
    app_state.messages = []
  app_state.messages.append(state_message)
  conversation = next(filter(
      lambda x: x.conversation_id == c.conversation_id,
      app_state.conversations), None)
  if conversation:
    conversation.message_ids.append(state_message.message_id)
  
  # Mark as processing
  app_state.background_tasks[message_id] = "processing"
  
  try:
    # Wait for the agent to complete processing
    response = await SendMessage(request)
    
    # Add agent response to messages
    if response:
      agent_message = convert_message_to_state(response)
      app_state.messages.append(agent_message)
      if conversation:
        conversation.message_ids.append(agent_message.message_id)
    
    # Mark as completed
    app_state.background_tasks[message_id] = "completed"
    
  except Exception as e:
    print(f"Error processing message: {e}")
    app_state.background_tasks[message_id] = "error"
  
  # Clear the input after successful send
  state.message_content = ""

async def send_message_enter(e: me.InputEnterEvent):
    """send message handler"""
    state = me.state(PageState)
    app_state = me.state(AppState)
    
    # Don't send if already processing
    if any(status == "processing" for status in app_state.background_tasks.values()):
        return
    
    state.message_content = e.value
    message_id = str(uuid.uuid4())
    app_state.background_tasks[message_id] = "queued"
    yield
    await send_message(state.message_content, message_id)
    yield

async def send_message_button(e: me.ClickEvent):
    """send message button handler"""
    state = me.state(PageState)
    app_state = me.state(AppState)
    
    # Don't send if already processing
    if any(status == "processing" for status in app_state.background_tasks.values()):
        return
    
    message_id = str(uuid.uuid4())
    app_state.background_tasks[message_id] = "queued"
    yield
    await send_message(state.message_content, message_id)
    yield

@me.component
def conversation():
    """Conversation component"""
    page_state = me.state(PageState)
    app_state = me.state(AppState)
    
    # Check if agent is processing
    is_processing = any(status == "processing" for status in app_state.background_tasks.values())
    
    if "conversation_id" in me.query_params:
      page_state.conversation_id = me.query_params["conversation_id"]
      app_state.current_conversation_id = page_state.conversation_id
      
      # Trigger a refresh if we have a conversation ID but no messages loaded yet
      if page_state.conversation_id and len(app_state.messages) == 0:
          # The SSE stream in page_scaffold will handle loading the conversation data
          # We just need to make sure the current_conversation_id is set correctly
          pass
    
    with me.box(
        style=me.Style(
            display="flex",
            justify_content="space-between",
            flex_direction="column",
        )
    ):
      for message in app_state.messages:
        if is_form(message):
          render_form(message, app_state)
        elif form_sent(message, app_state):
          chat_bubble(StateMessage(
              message_id=message.message_id,
              role=message.role,
              content=[("Form submitted", "text/plain")]
          ), message.message_id)
        else:
          chat_bubble(message, message.message_id)

      # Show processing indicator
      if is_processing:
        with me.box(style=me.Style(padding=me.Padding.all(10))):
          me.text("Agent is analyzing your request...", style=me.Style(color="gray"))

      with me.box(
          style=me.Style(
              display="flex",
              flex_direction="row",
              gap=5,
              align_items="center",
              min_width=500,
              width="100%",
          )
      ):
        me.input(
            label="How can I help you?" if not is_processing else "Please wait...",
            value=page_state.message_content,
            disabled=is_processing,  # Disable input while processing
            on_blur=on_blur,
            on_enter=send_message_enter if not is_processing else None,
            style=me.Style(min_width="80vw"),
        )
        with me.content_button(
            type="flat",
            disabled=is_processing,  # Disable button while processing
            on_click=send_message_button if not is_processing else None,
        ):
            me.icon(icon="send")
