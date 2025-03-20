import os
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage, HumanMessage
from tempfile import NamedTemporaryFile
from loguru import logger
from openai import OpenAI

from app.settings import OpenAISettings
from app.services.graph.config import GraphState, CustomNode, Stream

# Consider moving API key assignment from environment to dependency injection
os.environ["OPENAI_API_KEY"] = OpenAISettings.api_key


class AudioTranscription(Runnable):
    def __init__(self):

        self.client = OpenAI(api_key=OpenAISettings.api_key)
        self.model = "whisper-1"  # OpenAI's Whisper model

    def invoke(self, inputs: GraphState):
        # Check temp_audio_path in inputs
        if not inputs.temp_audio_path:
            raise ValueError("temp_audio_path not found in inputs")

        # Transcribe using OpenAI's Whisper API
        with open(inputs.temp_audio_path, "rb") as audio_file:
            transcription_response = self.client.audio.transcriptions.create(
                model=self.model, file=audio_file, language="en"
            )

        # # Clean up temporary file
        # if os.path.exists(inputs.temp_audio_path):
        #     os.unlink(inputs.temp_audio_path)

        transcription = transcription_response.text

        logger.info(f"Transcription: {transcription}")

        # Return transcription for further processing
        return {
            "user_question": transcription,
            "messages": [HumanMessage(content=transcription)],
            "transcription": transcription,
        }
