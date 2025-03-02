from rich.console import Console
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.services.graph.config import GraphState, Stream
from app.services.graph.tools.select_tools import ToolOptions
from app.services.graph.brain.prompts.brainstorming import CLARIFYING_QUESTIONS
from app.services.graph.brain.prompts.system_prompts import SYSTEM_PROMPT

console = Console()


class Brainstorming(Runnable):
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _manage_history(self, inputs: GraphState):
        """Manage the history of the conversation. With 6 pairs of messages."""
        if len(inputs.messages) > 6:
            inputs.messages = inputs.messages[-6:]
        return inputs

    def invoke(self, inputs: GraphState) -> Dict:
        """Invoke the brain module."""
        # TODO: Implement synchronous version if needed
        raise NotImplementedError("Method not implemented")

    async def ainvoke(self, inputs: GraphState) -> Dict:
        """Invoke the brain module asynchronously."""
        self._manage_history(inputs)

        # Create the prompt
        prompt = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + inputs.messages
            + [SystemMessage(content=CLARIFYING_QUESTIONS)]
            + [HumanMessage(content=inputs.user_question)]
        )
        # Invoke the LLM
        llm = self.llm.bind_tools([ToolOptions], tool_choice="auto")
        llm = llm.with_config({"tags": [Stream.stream.name]})
        console.rule("Answer Streaming")
        response = await llm.ainvoke(prompt)

        # Preprocess tool output
        if isinstance(response, AIMessage) and response.tool_calls:
            return {"tool_in_message": [response]}

        return {
            "assistant_response": response.content,
            "messages": [response],
        }
