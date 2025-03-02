from typing import Dict
from rich.console import Console
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.services.graph.brain.prompts.system_prompts import SYSTEM_PROMPT
from app.services.graph.config import GraphState, Stream


console = Console()


class ResponseAnswer(Runnable):
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _manage_history(self, inputs: GraphState):
        """Manage the history of the conversation. With 6 pairs of messages."""
        messages = []
        if len(inputs.messages) > 6:
            inputs.messages = inputs.messages[-6:]
        for i in inputs.messages:
            if isinstance(i, HumanMessage):
                messages.append(i)
            elif isinstance(i, AIMessage):
                messages.append(AIMessage(content=i.content))
        return messages

    def invoke(self, inputs: GraphState) -> Dict:
        """Invoke the brain module."""
        raise NotImplementedError("Method not implemented")

    async def ainvoke(self, inputs: GraphState) -> Dict:
        """Invoke the brain module asynchronously."""

        # Create the prompt
        prompt = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + self._manage_history(inputs)
            + [HumanMessage(content=inputs.user_question)]
        )
        # Invoke the LLM
        llm = self.llm.with_config({"tags": [Stream.stream.name]})
        response = await llm.ainvoke(prompt)

        return {
            "assistant_response": response.content,
            "messages": [response],
        }
