import time
from typing import List, Dict, Any
from psycopg_pool import AsyncConnectionPool
from langgraph.graph import StateGraph, START, END
from langchain_openai import OpenAI, AzureOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import AnyMessage

# Custom imports
from app.services.graph.config import GraphState, NodeState

# from app.services.graph.eyes.eye import Eye
# from app.services.graph.brain.nodes.response_anwser import ResponseAnswer
from app.services.graph.brain.nodes.brain import Brain
from app.services.graph.brain.nodes.brainstorming import Brainstorming
from app.services.graph.brain.nodes.refine_query_for_gg import GoogleSearch
from app.services.graph.brain.nodes.refine_answer_from_gg import RefineAnswerFromGG
from app.services.graph.brain.nodes.generate_sketch_concept import (
    GenerateSketchConcepts,
)
from app.services.graph.ears.hear import AudioTranscription

from app.services.graph.mouth.mouth import Mouth
from app.services.graph.hand.draw import Draw
from app.settings import PostgresSettings

# Connection pool configuration
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}


class AssistantAudioGraph:
    def __init__(self, llm: OpenAI | AzureOpenAI):
        self.llm = llm
        self.graph = StateGraph(GraphState)

        # Add edges for AI caller
        self.compiled_graph_for_ai_caller = self.compile_graph_for_ai_caller()

    def compile_graph(self):
        """Compile the graph"""
        # Add nodes and edges to the graph
        self._add_node()

        # Add edges to the graph
        self._add_edge()

        # Compile the graph
        return self.graph.compile()

    def compile_graph_for_ai_caller(self):
        """Compile the graph for AI caller"""
        # Add nodes and edges to the graph
        self._add_node()

        # Add edges to the graph
        self._add_edge_for_AI_caller()

        # Compile the graph
        return self.graph.compile()

    def _add_node(self):
        # ==================
        # Demo AI caller
        # ==================

        # Add audio transcription node
        self.graph.add_node(NodeState.hear_node.name, AudioTranscription().invoke)

        # Add brain node for answer audio
        self.graph.add_node(NodeState.thinking_node.name, Brain(self.llm).invoke)

        # Add mouth node
        self.graph.add_node(NodeState.mouth_node.name, Mouth().invoke)

    def _add_edge_for_AI_caller(self):
        # Start -> Audio Transcription
        self.graph.add_edge(START, NodeState.hear_node.name)
        # Audio Transcription -> Brain for Audio
        self.graph.add_edge(
            NodeState.hear_node.name,
            NodeState.thinking_node.name,
        )
        # Brain for Audio -> Response
        self.graph.add_edge(NodeState.thinking_node.name, NodeState.mouth_node.name)
        # Mouth -> End
        self.graph.add_edge(NodeState.mouth_node.name, END)

    def aai_caller(self, inputs: Dict[str, str], thread_id: str):
        """AI caller"""
        config = self._graph_config()
        config["configurable"] = {"thread_id": thread_id}
        init_state = self._init_state_for_ai_caller(
            audio_path=inputs["temp_audio_path"]
        )

        return self.compiled_graph_for_ai_caller.invoke(init_state, config=config)

    def _init_state_for_ai_caller(
        self,
        audio_path: str,
        messages: List[AnyMessage] = [],
    ):
        """Initialize the state for AI caller"""
        return GraphState(
            temp_audio_path=audio_path,
            user_question="",
            audio_path="",  # Nona
            messages=messages,
            tool_in_message=[],
            selected_tool={},
            detailed_description="",
            google_search_result="",
        )

    def _visualize(self):
        """Visualize the graph"""
        return self.compiled_graph_for_ai_caller.get_graph().draw_mermaid()

    def _graph_config(
        self,
        metadata: Dict[Any, str] = None,
        tags: List[str] = ["assistant"],
        callbacks: Dict[str, Any] = None,
        max_concurrency: int = 20,
        recursion_limit: int = 50,
    ):
        """Configure the graph"""
        return RunnableConfig(
            metadata=metadata,
            tags=tags,
            callbacks=callbacks,
            max_concurrency=max_concurrency,
            recursion_limit=recursion_limit,
        )
