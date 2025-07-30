from dataclasses import dataclass
from typing import Any, Callable

import mesop.labs as mel

@dataclass
class SSEEvent:
    data: dict[str, Any]

@mel.web_component(path="./sse_stream.js")
def sse_stream(
    *,
    url: str,
    trigger_event: Callable[[mel.WebEvent], Any],
    key: str | None = None,
):
    """Creates an SSE stream component that connects to a server-sent events endpoint.
    
    This component establishes a persistent connection to the specified URL and
    streams real-time updates to the UI. When events are received, the trigger_event
    callback is invoked with the parsed event data.
    
    Args:
        url: The SSE endpoint URL to connect to
        trigger_event: Callback function to handle incoming events
        key: Optional component key for React-like reconciliation
    
    Returns:
        The web component that was created.
    """
    return mel.insert_web_component(
        name="sse-stream-component",
        key=key,
        events={
            "triggerEvent": trigger_event,
        },
        properties={
            "url": url,
        },
    )