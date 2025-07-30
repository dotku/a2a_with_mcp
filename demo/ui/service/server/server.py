import asyncio
import base64
import threading
import os
import uuid
import json
from typing import Any, AsyncGenerator
from fastapi import APIRouter
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from common.types import Message, Task, FilePart, FileContent
from .in_memory_manager import InMemoryFakeAgentManager
from .application_manager import ApplicationManager
from .adk_host_manager import ADKHostManager, get_message_id
from service.types import (
    Conversation,
    Event,
    CreateConversationResponse,
    ListConversationResponse,
    SendMessageResponse,
    MessageInfo,
    ListMessageResponse,
    PendingMessageResponse,
    ListTaskResponse,
    RegisterAgentResponse,
    ListAgentResponse,
    GetEventResponse
)

class ConversationServer:
  """ConversationServer is the backend to serve the agent interactions in the UI

  This defines the interface that is used by the Mesop system to interact with
  agents and provide details about the executions.
  """
  def __init__(self, router: APIRouter):
    agent_manager = os.environ.get("A2A_HOST", "ADK")
    self.manager: ApplicationManager
    
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    uses_vertex_ai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE"
    
    if agent_manager.upper() == "ADK":
      print('use ADKHostManager')
      self.manager = ADKHostManager(api_key=api_key, uses_vertex_ai=uses_vertex_ai)
    else:
      print('use InMemoryFakeAgentManager')
      self.manager = InMemoryFakeAgentManager()
    self._file_cache = {} # dict[str, FilePart] maps file id to message data
    self._message_to_cache = {} # dict[str, str] maps message id to cache id

    router.add_api_route(
        "/conversation/create",
        self._create_conversation,
        methods=["POST"])
    router.add_api_route(
        "/conversation/list",
        self._list_conversation,
        methods=["POST"])
    router.add_api_route(
        "/message/send",
        self._send_message,
        methods=["POST"])
    router.add_api_route(
        "/events/get",
        self._get_events,
        methods=["POST"])
    router.add_api_route(
        "/message/list",
        self._list_messages,
        methods=["POST"])
    router.add_api_route(
        "/message/pending",
        self._pending_messages,
        methods=["POST"])
    router.add_api_route(
        "/task/list",
        self._list_tasks,
        methods=["POST"])
    router.add_api_route(
        "/agent/register",
        self._register_agent,
        methods=["POST"])
    router.add_api_route(
        "/agent/list",
        self._list_agents,
        methods=["POST"])
    router.add_api_route(
        "/message/file/{file_id}",
        self._files,
        methods=["GET"])
    router.add_api_route(
        "/api_key/update",
        self._update_api_key,
        methods=["POST"])
    router.add_api_route(
        "/events/stream",
        self._stream_events,
        methods=["GET"])

  # Update API key in manager
  def update_api_key(self, api_key: str):
    if isinstance(self.manager, ADKHostManager):
      self.manager.update_api_key(api_key)

  def _create_conversation(self):
    c = self.manager.create_conversation()
    return CreateConversationResponse(result=c)

  async def _send_message(self, request: Request):
    message_data = await request.json()
    message = Message(**message_data['params'])
    message = self.manager.sanitize_message(message)
    print(f"Sending message: {message.model_dump_json(indent=2)}")
    t = threading.Thread(target=lambda: asyncio.run(self.manager.process_message(message)))
    t.start()
    return SendMessageResponse(result=MessageInfo(
        message_id=message.metadata['message_id'],
        conversation_id=message.metadata['conversation_id'] if 'conversation_id' in message.metadata else '',
    ))

  async def _list_messages(self, request: Request):
    message_data = await request.json()
    conversation_id = message_data['params']
    conversation = self.manager.get_conversation(conversation_id)
    if conversation:
      return ListMessageResponse(result=self.cache_content(
          conversation.messages))
    return ListMessageResponse(result=[])

  def cache_content(self, messages: list[Message]):
    rval = []
    for m in messages:
      message_id = get_message_id(m)
      if not message_id:
        rval.append(m)
        continue
      new_parts = []
      for i, part in enumerate(m.parts):
        if part.type != 'file':
          new_parts.append(part)
          continue
        message_part_id = f"{message_id}:{i}"
        if message_part_id in self._message_to_cache:
          cache_id = self._message_to_cache[message_part_id]
        else:
          cache_id = str(uuid.uuid4())
          self._message_to_cache[message_part_id] = cache_id
        # Replace the part data with a url reference
        new_parts.append(FilePart(
            file=FileContent(
                mimeType=part.file.mimeType,
                uri=f"/message/file/{cache_id}",
            )
        ))
        if cache_id not in self._file_cache:
          self._file_cache[cache_id] = part
      m.parts = new_parts
      rval.append(m)
    return rval

  async def _pending_messages(self):
    return PendingMessageResponse(result=self.manager.get_pending_messages())

  def _list_conversation(self):
    return ListConversationResponse(result=self.manager.conversations)

  def _get_events(self):
    return GetEventResponse(result=self.manager.events)

  def _list_tasks(self):
    return ListTaskResponse(result=self.manager.tasks)

  async def _register_agent(self, request: Request):
    message_data = await request.json()
    url = message_data['params']
    self.manager.register_agent(url)
    return RegisterAgentResponse()

  async def _list_agents(self):
    return ListAgentResponse(result=self.manager.agents)

  def _files(self, file_id):
    if file_id not in self._file_cache:
      raise Exception("file not found")
    part = self._file_cache[file_id]
    if "image" in part.file.mimeType:
      return Response(
          content=base64.b64decode(part.file.bytes),
          media_type=part.file.mimeType)
    return Response(content=part.file.bytes, media_type=part.file.mimeType)
  
  async def _update_api_key(self, request: Request):
    """Update the API key"""
    try:
        data = await request.json()
        api_key = data.get("api_key", "")
        
        if api_key:
            # Update in the manager
            self.update_api_key(api_key)
            return {"status": "success"}
        return {"status": "error", "message": "No API key provided"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

  async def _stream_events(self) -> StreamingResponse:
    """Stream events using Server-Sent Events"""
    
    async def event_generator() -> AsyncGenerator[str, None]:
        last_event_count = 0
        last_message_count = {}  # conversation_id -> message_count
        last_task_count = 0
        last_pending_count = 0
        
        heartbeat_counter = 0
        while True:
            try:
                has_updates = False
                
                # Check for new events
                current_events = self.manager.events
                if isinstance(current_events, list) and len(current_events) > last_event_count:
                    for event in current_events[last_event_count:]:
                        yield f"data: {json.dumps({'type': 'event', 'data': event.model_dump()})}\n\n"
                    last_event_count = len(current_events)
                    has_updates = True
                
                # Check for new messages in conversations
                if isinstance(self.manager.conversations, list):
                    for conversation in self.manager.conversations:
                        conv_id = conversation.conversation_id
                        if hasattr(conversation, 'messages') and isinstance(conversation.messages, list):
                            current_msg_count = len(conversation.messages)
                            last_count = last_message_count.get(conv_id, 0)
                            
                            if current_msg_count > last_count:
                                # Send new messages
                                new_messages = conversation.messages[last_count:]
                                cached_messages = self.cache_content(new_messages)
                                for msg in cached_messages:
                                    yield f"data: {json.dumps({'type': 'message', 'conversation_id': conv_id, 'data': msg.model_dump()})}\n\n"
                                last_message_count[conv_id] = current_msg_count
                                has_updates = True
                
                # Check for new tasks only if there might be updates
                current_tasks = self.manager.tasks
                if isinstance(current_tasks, list) and len(current_tasks) > last_task_count:
                    for task in current_tasks[last_task_count:]:
                        yield f"data: {json.dumps({'type': 'task', 'data': task.model_dump()})}\n\n"
                    last_task_count = len(current_tasks)
                    has_updates = True
                
                # Check for pending message updates
                current_pending = self.manager.get_pending_messages()
                # Convert to dict format if it's a list of tuples
                if isinstance(current_pending, list) and current_pending and isinstance(current_pending[0], tuple):
                    pending_dict = dict(current_pending)
                else:
                    pending_dict = current_pending if isinstance(current_pending, dict) else {}
                
                current_pending_count = len(pending_dict) if isinstance(pending_dict, dict) else 0
                if current_pending_count != last_pending_count:
                    yield f"data: {json.dumps({'type': 'pending', 'data': pending_dict})}\n\n"
                    last_pending_count = current_pending_count
                    has_updates = True
                
                # Send heartbeat every 100 iterations (10 seconds at 0.1s intervals)
                heartbeat_counter += 1
                if heartbeat_counter >= 100:
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
                    heartbeat_counter = 0
                
                # Sleep longer if no updates to reduce resource usage
                if has_updates:
                    await asyncio.sleep(0.1)  # Quick check if we had updates
                else:
                    await asyncio.sleep(1.0)  # Longer sleep if no activity
                
            except Exception as e:
                print(f"SSE stream error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                await asyncio.sleep(5)  # Wait longer on error
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )
