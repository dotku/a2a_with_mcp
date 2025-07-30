import mesop as me
import json

from components.header import header
from components.page_scaffold import page_scaffold
from components.page_scaffold import page_frame
from state.state import AppState

def app_state_page(app_state: AppState):
    """App State Page"""
    with page_scaffold():  # pylint: disable=not-context-manager
        with page_frame():
          with header("App State", "memory"): pass
          
          with me.box(
              style=me.Style(
                  padding=me.Padding.all(16),
                  margin=me.Margin.all(8),
                  background="#f5f5f5",
                  border_radius=8,
                  font_family="monospace",
                  font_size=14,
                  white_space="pre-wrap",
                  overflow="auto",
                  max_height="600px"
              )
          ):
              try:
                  # Try to format as JSON if possible
                  state_dict = vars(app_state) if hasattr(app_state, '__dict__') else str(app_state)
                  formatted_state = json.dumps(state_dict, indent=2, default=str)
                  me.text(formatted_state)
              except Exception:
                  # Fallback to string representation
                  me.text(str(app_state))
          
          with me.box(
              style=me.Style(
                  font_size=12,
                  color="#888",
              )
          ):
              me.text("* Either `uses_vertex_ai` or `api_key` is set, or both are unset. This is used to determine if the app is using Vertex AI or not.")
