from pydantic import BaseModel
from enum import Enum, auto
from langchain_core.messages import AnyMessage, AIMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from typing import List, Any, Dict


class GraphState(BaseModel):
    user_question: str = ""
    audio_path: Any = ""
    assistant_response: str = ""
    messages: Annotated[List[AnyMessage], add_messages]
    tool_in_message: List[AnyMessage | Any] = []
    selected_tool: Dict = {}  # Nona
    detailed_description: str = ""
    google_search_result: str = ""


class NodeState(Enum):
    eye_node = auto()
    brain_node = auto()
    mouth_node = auto()
    hand_node = auto()
    sketch_ideas_node = auto()
    google_search_node = auto()
    refine_answer_from_gg_node = auto()

    # Response nodes
    response_node = auto()


class Stream(Enum):
    stream = auto()
    stream_complete = auto()


class ToolActions(Enum):
    use_hand_to_draw = auto()
    use_mouth_to_speak = auto()
    use_google_to_search = auto()
    use_brainstorming = auto()
    no_action = auto()
    next_action = auto()


class CustomNode(Enum):
    on_image_generated = auto()
    on_audio_generated = auto()
