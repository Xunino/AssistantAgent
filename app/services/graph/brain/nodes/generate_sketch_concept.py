from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.services.graph.brain.prompts.sketch_ideas import SKETCH_CONCEPT_PROMPT
from app.services.graph.config import GraphState


class GenerateSketchConcepts(Runnable):
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _history(self, inputs: GraphState):
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
        prompt = (
            [SystemMessage(content=SKETCH_CONCEPT_PROMPT)]
            + self._history(inputs)
            + [HumanMessage(content=inputs.user_question)]
        )
        response = self.llm.invoke(prompt)
        return {
            "detailed_description": response.content,
            "messages": [response],
        }

    async def ainvoke(self, inputs: GraphState) -> Dict:
        """Invoke the brain module asynchronously."""
        # TODO: Implement asynchronous version
        raise NotImplementedError("Method not implemented")
