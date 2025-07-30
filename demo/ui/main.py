"""A UI solution and host service to interact with the agent framework.
run:
  uv main.py
"""
import sys
import os # For path manipulation

import asyncio
import threading

import mesop as me

from state.state import AppState
from components.page_scaffold import page_scaffold
from components.api_key_dialog import api_key_dialog
from pages.home import home_page_content
from pages.agent_list import agent_list_page
from pages.conversation import conversation_page
from pages.event_list import event_list_page
from pages.settings import settings_page_content
from pages.task_list import task_list_page
from pages.app_state import app_state_page
from state import host_agent_service
from service.server.server import ConversationServer
from fastapi import FastAPI, APIRouter
from fastapi.middleware.wsgi import WSGIMiddleware
from dotenv import load_dotenv

# --- Start of sys.path modification ---
# Add the project root to sys.path to allow for absolute imports from project root
# This script is located at <project_root>/demo/ui/main.py
_main_py_file_abs_path = os.path.abspath(__file__)
_ui_dir_abs_path = os.path.dirname(_main_py_file_abs_path)
_demo_dir_abs_path = os.path.dirname(_ui_dir_abs_path)
# _project_root_abs_path should now be the <project_root> directory
_project_root_abs_path = os.path.dirname(_demo_dir_abs_path)

if _project_root_abs_path not in sys.path:
    sys.path.insert(0, _project_root_abs_path) # Use insert(0, ...) for higher precedence

print(f"DEBUG: Running from __file__: {__file__}")
print(f"DEBUG: Absolute path of __file__: {_main_py_file_abs_path}")
print(f"DEBUG: Calculated project root to add to sys.path: {_project_root_abs_path}")
print(f"DEBUG: Current sys.path after modification: {sys.path}")
# --- End of sys.path modification ---

load_dotenv()

def _load_conversation_data(state, conversation_id):
    """Load conversation data in a background thread"""
    async def _async_load():
        await host_agent_service.UpdateAppState(state, conversation_id)
    
    # Run the async function in a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_async_load())
    finally:
        loop.close()

def on_load(e: me.LoadEvent):  # pylint: disable=unused-argument
    """On load event"""
    state = me.state(AppState)
    me.set_theme_mode(state.theme_mode)
    if "conversation_id" in me.query_params:
      state.current_conversation_id = me.query_params["conversation_id"]
      # Load conversation data in background thread
      if state.current_conversation_id:
          threading.Thread(
              target=_load_conversation_data,
              args=(state, state.current_conversation_id),
              daemon=True
          ).start()
    else:
      state.current_conversation_id = ""
    
    # check if the API key is set in the environment
    # and if the user is using Vertex AI
    uses_vertex_ai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE"
    api_key = os.getenv("GOOGLE_API_KEY", "")
    
    if uses_vertex_ai:
        state.uses_vertex_ai = True
    elif api_key:
        state.api_key = api_key
    else:
        # Show the API key dialog if both are not set
        state.api_key_dialog_open = True

# Policy to allow the lit custom element to load
security_policy=me.SecurityPolicy(
    allowed_script_srcs=[
      'https://cdn.jsdelivr.net',
    ]
  )


@me.page(
    path="/app_state",
    title="App State",
    on_load=on_load,
    security_policy=security_policy,
)
def get_app_state_page():
    """App State Page"""
    api_key_dialog()
    app_state_page(me.state(AppState))

@me.page(
    path="/",
    title="Chat",
    on_load=on_load,
    security_policy=security_policy,
)
def home_page():
    """Main Page"""
    state = me.state(AppState)
    # Show API key dialog if needed
    api_key_dialog()
    with page_scaffold():  # pylint: disable=not-context-manager
        home_page_content(state)


@me.page(
    path="/agents",
    title="Agents",
    on_load=on_load,
    security_policy=security_policy,
)
def another_page():
    """Another Page"""
    api_key_dialog()
    agent_list_page(me.state(AppState))


@me.page(
    path="/conversation",
    title="Conversation",
    on_load=on_load,
    security_policy=security_policy,
)
def chat_page():
    """Conversation Page."""
    api_key_dialog()
    conversation_page(me.state(AppState))

@me.page(
    path="/event_list",
    title="Event List",
    on_load=on_load,
    security_policy=security_policy,
)
def event_page():
    """Event List Page."""
    api_key_dialog()
    event_list_page(me.state(AppState))


@me.page(
    path="/settings",
    title="Settings",
    on_load=on_load,
    security_policy=security_policy,
)
def settings_page():
    """Settings Page."""
    api_key_dialog()
    settings_page_content()


@me.page(
    path="/task_list",
    title="Task List",
    on_load=on_load,
    security_policy=security_policy,
)
def task_page():
    """Task List Page."""
    api_key_dialog()
    task_list_page(me.state(AppState))

# Setup the server global objects
app = FastAPI()
router = APIRouter()
agent_server = ConversationServer(router)
app.include_router(router)

app.mount(
    "/",
    WSGIMiddleware(
        me.create_wsgi_app(debug_mode=os.environ.get("DEBUG_MODE", "") == "true")
    ),
)

if __name__ == "__main__":    

    import uvicorn
    # Setup the connection details, these should be set in the environment
    host = os.environ.get("A2A_UI_HOST", "0.0.0.0")
    port = int(os.environ.get("A2A_UI_PORT", "12000"))

    # Set the client to talk to the server
    host_agent_service.server_url = f"http://{host}:{port}"

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        reload_includes=["*.py", "*.js"],
        timeout_graceful_shutdown=0,
    )
