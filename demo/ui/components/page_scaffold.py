import mesop as me
import mesop.labs as mel
import asyncio

from .side_nav import sidenav
from .sse_stream import sse_stream

from state.state import AppState
from state.host_agent_service import UpdateAppState

from styles.styles import (
    MAIN_COLUMN_STYLE,
    PAGE_BACKGROUND_PADDING_STYLE,
    PAGE_BACKGROUND_STYLE,
    SIDENAV_MAX_WIDTH,
    SIDENAV_MIN_WIDTH,
)

async def refresh_app_state(e: mel.WebEvent):  # pylint: disable=unused-argument
    """Refresh app state event handler"""
    yield
    app_state = me.state(AppState)
    await UpdateAppState(app_state, app_state.current_conversation_id)
    yield


async def handle_global_sse_event(e):
    """Handle global SSE events for app state updates"""
    yield
    app_state = me.state(AppState)
    
    if not hasattr(e, 'value') or not e.value:
        yield
        return
    
    try:
        import json
        # Handle the nested value structure from Mesop
        if isinstance(e.value, dict) and 'value' in e.value:
            # Extract the inner value and parse as JSON
            inner_value = e.value['value']
            if isinstance(inner_value, str):
                event_data = json.loads(inner_value)
            else:
                event_data = inner_value
        elif isinstance(e.value, dict):
            event_data = e.value
        else:
            # Fallback: try to parse as JSON string
            event_data = json.loads(str(e.value))
    except (json.JSONDecodeError, AttributeError, TypeError) as error:
        print(f"Global SSE event parsing error: {error}, value type: {type(e.value)}, value: {e.value}")
        yield
        return
    
    event_type = event_data.get('type')
    
    # Handle all SSE events globally
    if event_type == 'message':
        # New message received - trigger a full refresh to update conversations
        await UpdateAppState(app_state, app_state.current_conversation_id)
    
    elif event_type == 'task':
        # Task updates should trigger a refresh
        await UpdateAppState(app_state, app_state.current_conversation_id)
    
    elif event_type == 'pending':
        # Update pending messages globally
        pending_data = event_data.get('data', {})
        app_state.background_tasks = pending_data
    
    elif event_type == 'heartbeat':
        # Keep connection alive, no action needed
        pass
    
    elif event_type == 'event':
        # New events should trigger a refresh
        await UpdateAppState(app_state, app_state.current_conversation_id)
    
    yield


@me.content_component
def page_scaffold():
    """page scaffold component"""

    app_state = me.state(AppState)
    
    # Use SSE instead of polling for real-time updates
    sse_stream(
        url="http://localhost:12000/events/stream",
        trigger_event=handle_global_sse_event,
        key="global_sse"
    )
    

    sidenav("")

    with me.box(
        style=me.Style(
            display="flex",
            flex_direction="column",
            height="100%",
            margin=me.Margin(
                left=SIDENAV_MAX_WIDTH if app_state.sidenav_open else SIDENAV_MIN_WIDTH,
            ),
        ),
    ):
        with me.box(
            style=me.Style(
                background=me.theme_var("background"),
                height="100%",
                overflow_y="scroll",
                margin=me.Margin(bottom=20),
            )
        ):
            me.slot()


@me.content_component
def page_frame():
    """Page Frame"""
    with me.box(style=MAIN_COLUMN_STYLE):
        with me.box(style=PAGE_BACKGROUND_STYLE):
            with me.box(style=PAGE_BACKGROUND_PADDING_STYLE):
                me.slot()
