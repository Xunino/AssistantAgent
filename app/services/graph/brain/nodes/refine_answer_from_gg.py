from datetime import datetime
from rich.console import Console
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.services.graph.brain.prompts.system_prompts import SYSTEM_PROMPT
from app.services.graph.config import GraphState, Stream

console = Console()


class RefineAnswerFromGG(Runnable):
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _history(self, inputs: GraphState):
        """Manage the history of the conversation."""
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
        raise NotImplementedError("Method not implemented")

    def _get_prompt(self, inputs: GraphState) -> str:
        """Get the prompt for the LLM."""
        return f"""Here is the search result from Google for question `{inputs.user_question}`:\n<google_search_result>{inputs.google_search_result}\n</google_search_result>"""

    async def ainvoke(self, inputs: GraphState) -> Dict:
        """Invoke the brain module asynchronously."""

        # Create the prompt
        prompt = (
            [
                SystemMessage(
                    content="Today is " + datetime.now().strftime("%Y-%m-%d")
                ),
                SystemMessage(content=SYSTEM_PROMPT),
            ]
            + self._history(inputs)
            + [
                AIMessage(content=self._get_prompt(inputs)),
                HumanMessage(content=inputs.user_question),
            ]
        )
        # Invoke the LLM
        llm = self.llm.with_config({"tags": [Stream.stream.name]})
        # Optionally use logger.info here instead of console.rule if preferred
        response = await llm.ainvoke(prompt)

        return {
            "assistant_response": response.content,
            "messages": [response],
        }
