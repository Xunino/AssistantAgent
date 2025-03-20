import time
from typing import List, Dict, Any
from psycopg_pool import AsyncConnectionPool
from langgraph.graph import StateGraph, START, END
from langchain_openai import OpenAI, AzureOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import AnyMessage

# Custom imports
from app.services.graph.config import (
    GraphState,
    NodeState,
    ToolActions,
    Stream,
    CustomNode,
)

# from app.services.graph.eyes.eye import Eye
# from app.services.graph.brain.nodes.response_anwser import ResponseAnswer
# from app.services.graph.brain.nodes.brain import Brain
from app.services.graph.brain.nodes.brainstorming import Brainstorming
from app.services.graph.brain.nodes.refine_query_for_gg import GoogleSearch
from app.services.graph.brain.nodes.refine_answer_from_gg import RefineAnswerFromGG
from app.services.graph.brain.nodes.generate_sketch_concept import (
    GenerateSketchConcepts,
)

# from app.services.graph.mouth.mouth import Mouth
from app.services.graph.hand.draw import Draw
from app.settings import PostgresSettings

# Connection pool configuration
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}


class AssistantGraph:
    def __init__(self, llm: OpenAI | AzureOpenAI):
        self.llm = llm
        self.graph = StateGraph(GraphState)

        # Compile the graph
        self.compiled_graph = self.compile_graph()

    def compile_graph(self):
        """Compile the graph"""
        # Add nodes and edges to the graph
        self._add_node()

        # Add edges to the graph
        self._add_edge()

        # Compile the graph
        return self.graph.compile()

    def _add_node(self):
        # Add eye node
        # self.graph.add_node(NodeState.eye_node.name, Eye().ainvoke)

        # Add brain node
        self.graph.add_node(NodeState.brain_node.name, Brainstorming(self.llm).ainvoke)

        # Add tool selection node
        # self.graph.add_node(
        #     NodeState.tool_selection_node.name, ToolSelection(self.llm).invoke
        # )

        # # Add mouth node
        # self.graph.add_node(NodeState.mouth_node.name, Mouth().invoke)

        # Add hand node
        self.graph.add_node(NodeState.hand_node.name, Draw().ainvoke)

        # Add sketch ideas node
        self.graph.add_node(
            NodeState.sketch_ideas_node.name, GenerateSketchConcepts(self.llm).invoke
        )

        # Add google search node
        self.graph.add_node(
            NodeState.google_search_node.name, GoogleSearch(self.llm).ainvoke
        )

        # Add refine answer from gg node
        self.graph.add_node(
            NodeState.refine_answer_from_gg_node.name,
            RefineAnswerFromGG(self.llm).ainvoke,
        )

        # # Add response node
        # self.graph.add_node(
        #     NodeState.talking_node.name, ResponseAnswer(self.llm).ainvoke
        # )

    def _add_edge(self):
        # Add edges to the graph

        # Start -> Brain
        self.graph.add_edge(START, NodeState.brain_node.name)

        # Brain -> Decision
        self.graph.add_conditional_edges(
            NodeState.brain_node.name,
            self._process_decision,
            {
                ToolActions.use_hand_to_draw.name: NodeState.sketch_ideas_node.name,
                ToolActions.no_action.name: END,
                ToolActions.use_google_to_search.name: NodeState.google_search_node.name,
            },
        )

        # Sketch Ideas -> Hand
        self.graph.add_edge(NodeState.sketch_ideas_node.name, NodeState.hand_node.name)

        # Hand -> Response
        self.graph.add_edge(NodeState.hand_node.name, END)

        # Google Search -> Refine Answer From GG
        self.graph.add_edge(
            NodeState.google_search_node.name, NodeState.refine_answer_from_gg_node.name
        )

        # Refine Answer From GG -> Response
        self.graph.add_edge(NodeState.refine_answer_from_gg_node.name, END)

    async def astream_with_checkpointer(
        self,
        inputs: Dict[str, str],
        thread_id: str,
        version: str = "v2",
    ):
        """Stream the graph"""
        # Configure the graph
        config = self._graph_config()
        config["configurable"] = {"thread_id": thread_id}

        # Initialize the state
        init_state = self._init_state(
            user_question=inputs["user_question"],
            messages=inputs["messages"],
        )

        # Graph with checkpointer
        graph = self.graph

        # Create a new graph with checkpointer
        async with AsyncConnectionPool(
            conninfo=PostgresSettings.uri,
            max_size=20,
            kwargs=connection_kwargs,
        ) as pool:
            checkpointer = AsyncPostgresSaver(pool)

            # Setup the checkpointer
            await checkpointer.setup()

            # Compile the graph with checkpointer
            graph_with_checkpointer = graph.compile(checkpointer)

            # Stream the graph
            async for event in graph_with_checkpointer.astream_events(
                init_state,
                version=version,
                config=config,
            ):
                event_kind = event["event"]
                event_name = event["name"]
                tags = event.get("tags", [])

                # filter on the custom tag
                if event_kind == "on_chat_model_stream" and Stream.stream.name in tags:
                    data = event["data"]
                    if data["chunk"].content:
                        # Empty content in the context of OpenAI or Anthropic usually means
                        # that the model is asking for a tool to be invoked.
                        # So we only print non-empty content
                        yield data["chunk"].content

                elif event_name == CustomNode.on_image_generated.name:
                    if event_kind == "on_custom_event":
                        data = event["data"]
                        if data["chunk"]:
                            # Empty content in the context of OpenAI or Anthropic usually means
                            # that the model is asking for a tool to be invoked.
                            # So we only print non-empty content
                            for data in data["chunk"].content.split(" "):
                                time.sleep(0.5)
                                yield data + " "

    async def astream(
        self,
        inputs,
        thread_id: str,
        version: str = "v2",
    ):
        """Stream the graph"""
        config = self._graph_config()
        config["configurable"] = {"thread_id": thread_id}
        init_state = self._init_state(
            user_question=inputs["user_question"],
            messages=inputs["messages"],
        )

        async for event in self.compiled_graph.astream_events(
            init_state,
            version=version,
            config=config,
        ):
            event_kind = event["event"]
            event_name = event["name"]
            tags = event.get("tags", [])

            # filter on the custom tag
            if event_kind == "on_chat_model_stream" and Stream.stream.name in tags:
                data = event["data"]
                if data["chunk"].content:
                    # Empty content in the context of OpenAI or Anthropic usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content
                    yield data["chunk"].content

            elif event_name == CustomNode.on_image_generated.name:
                if event_kind == "on_custom_event":
                    data = event["data"]
                    if data["chunk"]:
                        # Empty content in the context of OpenAI or Anthropic usually means
                        # that the model is asking for a tool to be invoked.
                        # So we only print non-empty content
                        for data in data["chunk"].content.split("\n"):
                            time.sleep(0.5)
                            yield data["chunk"].content

    def _init_state(self, user_question: str, messages: List[AnyMessage] = []):
        """Initialize the state"""
        return GraphState(
            user_question=user_question,
            audio_path="",  # Nona
            messages=messages,
            tool_in_message=[],
            selected_tool={},
            detailed_description="",
            google_search_result="",
        )

    def _process_decision(self, inputs: GraphState):
        """Process the decision"""
        if inputs.tool_in_message:
            tool_in_message = inputs.tool_in_message[-1]
            if tool_in_message.tool_calls:
                if tool_in_message.tool_calls[0]["args"].get("use_hand_to_draw", False):
                    return ToolActions.use_hand_to_draw.name
                elif tool_in_message.tool_calls[0]["args"].get(
                    "use_google_to_search", False
                ):
                    return ToolActions.use_google_to_search.name

        return ToolActions.no_action.name

    def _visualize(self):
        """Visualize the graph"""
        return self.compiled_graph.get_graph().draw_mermaid()

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


if __name__ == "__main__":
    # Create a new graph
    graph = AssistantGraph(OpenAI())
    print(graph._visualize())
