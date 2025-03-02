import os
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage

from app.settings import OpenAISettings
from app.services.graph.config import GraphState, CustomNode, Stream

# Consider moving API key assignment from environment to dependency injection
os.environ["OPENAI_API_KEY"] = OpenAISettings.api_key


class Draw(Runnable):
    def __init__(self):
        self.client = DallEAPIWrapper()
        self.client.model_name = "dall-e-2"
        self.client.size = "256x256"

    def invoke(self, inputs: GraphState):
        # TODO: Implement synchronous invoke if needed
        raise NotImplementedError("Method not implemented")

    async def ainvoke(self, inputs: GraphState):
        """Invoke the draw module."""
        config = RunnableConfig(tags=[Stream.stream.name])
        image_url = self.client.run(inputs.detailed_description)

        message = f"Here is the link of image: {image_url}"
        await adispatch_custom_event(
            name=CustomNode.on_image_generated.name,
            data={
                "chunk": AIMessage(content=message),
                "assistant_response": "Link of image has been generated in chat. Please click on the link to view the image.",
            },
            config=config,
        )
        return {
            "messages": [AIMessage(content=message)],
            "assistant_response": "Link of image has been generated in chat. Please click on the link to view the image.",
        }
