from rich.console import Console
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.services.graph.config import GraphState, Stream
from app.services.graph.brain.prompts.system_prompts import SYSTEM_PROMPT_FOR_CALLER

console = Console()


class Brain(Runnable):
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _manage_history(self, inputs: GraphState):
        """Manage the history of the conversation. With 6 pairs of messages."""
        if len(inputs.messages) > 6:
            inputs.messages = inputs.messages[-6:]
        return inputs

    def invoke(self, inputs: GraphState) -> Dict:
        """Invoke the brain module."""
        self._manage_history(inputs)

        # Create the prompt
        prompt = [
            SystemMessage(content=SYSTEM_PROMPT_FOR_CALLER),
            HumanMessage(content=inputs.user_question),
        ]
        # Invoke the LLM
        response = self.llm.invoke(prompt)

        # Preprocess tool output
        if isinstance(response, AIMessage) and response.tool_calls:
            return {"tool_in_message": [response]}

        return {
            "assistant_response": response.content,
            "messages": [response],
        }

    def ainvoke(self, input, config=None, **kwargs):
        """Not implemented"""
        raise NotImplementedError("Method not implemented")
